"""Proximal Policy Optimization (clipped objective) with GAE.

The rollout itself lives with the caller (self-play trainer, CartPole sanity
check); this module holds the storage, the advantage computation and the
gradient update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .policy import ActorCritic


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int):
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logp = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.n_steps = n_steps

    def add(self, t, obs, actions, logp, values, rewards, dones):
        self.obs[t] = obs
        self.actions[t] = actions
        self.logp[t] = logp
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """``dones[t] == 1`` means the episode ended right after step ``t``."""
    n_steps = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_value)
    for t in reversed(range(n_steps)):
        next_v = last_value if t == n_steps - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * live - values[t]
        last = delta + gamma * lam * live * last
        adv[t] = last
    return adv, adv + values


class PPO:
    def __init__(self, net: ActorCritic, cfg: PPOConfig = PPOConfig()):
        self.net = net
        self.cfg = cfg
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def update(self, buf: RolloutBuffer, last_value: np.ndarray) -> dict:
        cfg = self.cfg
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, last_value, cfg.gamma, cfg.gae_lambda)

        obs = torch.from_numpy(buf.obs.reshape(-1, buf.obs.shape[-1]))
        actions = torch.from_numpy(buf.actions.reshape(-1))
        old_logp = torch.from_numpy(buf.logp.reshape(-1))
        adv_t = torch.from_numpy(adv.reshape(-1))
        ret_t = torch.from_numpy(ret.reshape(-1))

        batch = obs.shape[0]
        mb = batch // cfg.minibatches
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch)
            for start in range(0, batch - mb + 1, mb):
                idx = perm[start : start + mb]
                logp, entropy, value = self.net.evaluate(obs[idx], actions[idx])
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()

                a = adv_t[idx]
                if cfg.norm_adv:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * ((value - ret_t[idx]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["policy_loss"] += pg_loss.item()
                    stats["value_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                n_updates += 1

        stats = {k: v / n_updates for k, v in stats.items()}
        var_y = ret.var()
        stats["explained_var"] = float(1 - np.var(ret - buf.values) / var_y) if var_y > 0 else float("nan")
        return stats

"""Sanity check: our PPO should solve CartPole-v1 (mean return near 500)."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.policy import ActorCritic  # noqa: E402
from football.ppo import PPO, PPOConfig, RolloutBuffer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sanity_cartpole/metrics.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    envs = [gym.make("CartPole-v1") for _ in range(args.n_envs)]
    obs = np.stack([e.reset(seed=args.seed + i)[0] for i, e in enumerate(envs)]).astype(np.float32)

    net = ActorCritic(obs_dim=4, n_actions=2, hidden=64)
    ppo = PPO(net, PPOConfig(gamma=0.99, ent_coef=0.0, lr=2.5e-4))
    buf = RolloutBuffer(args.n_steps, args.n_envs, 4)

    ep_ret = np.zeros(args.n_envs)
    finished = []
    rows = []
    n_updates = args.total_steps // (args.n_envs * args.n_steps)
    t0 = time.time()
    for update in range(1, n_updates + 1):
        ppo.set_lr(ppo.cfg.lr * (1 - (update - 1) / n_updates))
        for t in range(args.n_steps):
            a, logp, v = net.act(torch.from_numpy(obs))
            a = a.numpy()
            next_obs, rew, done = [], np.zeros(args.n_envs), np.zeros(args.n_envs)
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(int(a[i]))
                rew[i], done[i] = r, term or trunc
                ep_ret[i] += r
                if done[i]:
                    finished.append(ep_ret[i])
                    ep_ret[i] = 0
                    o, _ = e.reset()
                next_obs.append(o)
            buf.add(t, obs, a, logp.numpy(), v.numpy(), rew, done)
            obs = np.stack(next_obs).astype(np.float32)
        last_v = net.value(torch.from_numpy(obs)).detach().numpy()
        stats = ppo.update(buf, last_v)
        recent = float(np.mean(finished[-20:])) if finished else 0.0
        steps = update * args.n_envs * args.n_steps
        rows.append({"env_steps": steps, "mean_return": recent, **stats})
        if update % 10 == 0 or update == n_updates:
            print(f"steps {steps:>7d}  return {recent:6.1f}  kl {stats['approx_kl']:.2e}  {time.time() - t0:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"final mean return (last 20 episodes): {np.mean(finished[-20:]):.1f}")


if __name__ == "__main__":
    main()

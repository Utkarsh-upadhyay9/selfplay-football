"""Actor-critic network, checkpoint I/O and a numpy-facing player wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import N_ACTIONS, OBS_DIM


def _layer(n_in: int, n_out: int, std: float = np.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Separate MLPs for the policy and the value function."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, n_actions, std=0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, std=1.0),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = Categorical(logits=self.actor(obs))
        action = dist.sample()
        return action, dist.log_prob(action), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = Categorical(logits=self.actor(obs))
        return dist.log_prob(actions), dist.entropy(), self.value(obs)

    def arch(self) -> dict:
        return {"obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden}


class PolicyPlayer:
    """Wrap a network as a ``obs (B, D) -> actions (B,)`` numpy callable."""

    def __init__(self, net: ActorCritic, greedy: bool = False, name: str = "policy"):
        self.net = net
        self.greedy = greedy
        self.name = name

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        logits = self.net.actor(torch.as_tensor(obs, dtype=torch.float32))
        if self.greedy:
            return logits.argmax(dim=-1).numpy()
        return Categorical(logits=logits).sample().numpy()


def save_checkpoint(path: str | Path, net: ActorCritic, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": net.arch(), "state_dict": net.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str | Path) -> tuple[ActorCritic, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(**data["arch"])
    net.load_state_dict(data["state_dict"])
    net.eval()
    return net, data.get("meta", {})


def load_player(path: str | Path, greedy: bool = False, name: str | None = None) -> PolicyPlayer:
    net, _ = load_checkpoint(path)
    return PolicyPlayer(net, greedy=greedy, name=name or Path(path).stem)

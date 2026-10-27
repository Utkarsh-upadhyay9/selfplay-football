"""Build a player from a short spec string: ``random``, ``chase`` or a checkpoint path."""

from __future__ import annotations

from pathlib import Path

from .bots import ChaseBot, RandomBot
from .policy import load_player


def make_player(spec: str, greedy: bool = False, seed: int | None = None):
    if spec == "random":
        return RandomBot(seed=seed)
    if spec == "chase":
        return ChaseBot()
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"unknown player spec {spec!r} (expected random, chase or a .pt file)")
    return load_player(path, greedy=greedy)

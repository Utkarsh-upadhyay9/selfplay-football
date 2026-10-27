"""Scripted opponents used as baselines.

Every "player" in this project (bots and learned policies) is a callable that
maps a batch of ego-centric observations ``(B, OBS_DIM)`` to integer actions
``(B,)``.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG, GameConfig, N_ACTIONS, N_MOVES
from .core import MOVE_DIRS


class RandomBot:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, N_ACTIONS, size=len(obs))


class ChaseBot:
    """Run to the spot behind the ball (relative to the opponent goal) and shoot.

    When the bot is lined up behind the ball and within kicking range it kicks
    towards the goal; otherwise it moves to the "behind the ball" spot, which
    makes it dribble the ball forward when it gets there.
    """

    name = "chase"

    def __init__(self, config: GameConfig = DEFAULT_CONFIG, align: float = 0.6):
        self.cfg = config
        self.align = align

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        c = self.cfg
        scale = np.array([0.5 * c.field_l, 0.5 * c.field_h])
        me = obs[:, 0:2] * scale
        ball = obs[:, 8:10] * scale
        goal = np.array([0.5 * c.field_l, 0.0])

        aim = goal - ball
        aim /= np.maximum(np.linalg.norm(aim, axis=1, keepdims=True), 1e-8)
        behind = ball - aim * (c.player_r + c.ball_r + 0.5)

        # If the bot is goal-side of the ball, walking straight to the spot
        # behind it would push the ball backwards, so detour around the side.
        perp = np.stack([-aim[:, 1], aim[:, 0]], axis=1)
        offset = me - ball
        ahead = (offset * aim).sum(axis=1) > -1.0
        side = np.where((offset * perp).sum(axis=1) >= 0, 1.0, -1.0)
        detour = ball - aim * 2.0 + perp * side[:, None] * (2.0 * c.player_r + c.ball_r + 2.0)
        behind = np.where(ahead[:, None], detour, behind)

        to_target = behind - me
        dist_target = np.linalg.norm(to_target, axis=1)
        # Seek the ball directly once lined up behind it so the bot dribbles.
        to_ball = ball - me
        dist_ball = np.linalg.norm(to_ball, axis=1)
        to_ball_u = to_ball / np.maximum(dist_ball, 1e-8)[:, None]
        lined_up = (to_ball_u * aim).sum(axis=1) > self.align
        heading = np.where((lined_up & (dist_ball < 3 * c.kick_range))[:, None], to_ball, to_target)

        move = np.argmax(heading @ MOVE_DIRS[1:].T, axis=1) + 1
        move[(dist_target < 0.5) & ~lined_up] = 0

        kick = (dist_ball <= c.kick_range) & lined_up
        return move + N_MOVES * kick.astype(np.int64)


BOTS = {"random": RandomBot, "chase": ChaseBot}


def make_bot(name: str, **kwargs):
    return BOTS[name](**kwargs)

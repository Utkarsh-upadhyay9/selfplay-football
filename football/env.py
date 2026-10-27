"""Gymnasium single-agent wrapper around `FootballCore`.

The agent controls one side; the other side is driven by ``opponent`` (any
callable from a batch of observations to a batch of actions). Training uses
`FootballCore` directly for speed, but this wrapper makes the game usable
with standard Gymnasium tooling.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bots import ChaseBot
from .config import DEFAULT_CONFIG, GameConfig, N_ACTIONS, OBS_DIM
from .core import FootballCore


class FootballEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        opponent=None,
        side: int = 0,
        shaping_coef: float = 0.0,
        config: GameConfig = DEFAULT_CONFIG,
        render_mode: str | None = None,
    ):
        assert side in (0, 1)
        self.cfg = config
        self.opponent = opponent if opponent is not None else ChaseBot(config)
        self.side = side
        self.shaping_coef = shaping_coef
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.core = FootballCore(1, config, auto_reset=False)
        self._obs = None
        self._renderer = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        self._obs = self.core.reset()
        return self._obs[0, self.side].copy(), {}

    def step(self, action):
        opp = 1 - self.side
        actions = np.zeros((1, 2), dtype=np.int64)
        actions[0, self.side] = int(action)
        actions[0, opp] = int(self.opponent(self._obs[:, opp])[0])
        self._obs, sparse, shaping, _, info = self.core.step(actions)
        reward = float(sparse[0, self.side] + self.shaping_coef * shaping[0, self.side])
        goal = int(info["goal"][0]) * (1 if self.side == 0 else -1)
        if self.render_mode == "human":
            self.render()
        return (
            self._obs[0, self.side].copy(),
            reward,
            bool(info["terminated"][0]),
            bool(info["truncated"][0]),
            {"goal": goal},
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .render import Renderer

            self._renderer = Renderer(self.cfg, headless=self.render_mode == "rgb_array")
        self._renderer.draw(self.core, 0)
        if self.render_mode == "human":
            self._renderer.flip()
            return None
        return self._renderer.rgb_array()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

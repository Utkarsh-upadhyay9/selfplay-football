"""Pygame renderer for a single game from a `FootballCore` batch."""

from __future__ import annotations

import os

import numpy as np

from .config import GameConfig

GRASS = (46, 125, 50)
GRASS_DARK = (40, 112, 45)
LINE = (235, 245, 235)
TEAM_COLORS = ((229, 57, 53), (30, 136, 229))
BALL = (250, 250, 250)
HUD = (20, 20, 20)


class Renderer:
    def __init__(self, config: GameConfig, scale: float = 8.0, headless: bool = False, hud_h: int = 36):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pg = pygame
        pygame.init()
        self.cfg = config
        self.scale = scale
        self.margin = int(3 * scale)
        self.hud_h = hud_h
        self.w = int(config.field_l * scale) + 2 * self.margin
        self.h = int(config.field_h * scale) + 2 * self.margin + hud_h
        if headless:
            self.screen = pygame.Surface((self.w, self.h))
        else:
            self.screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Self-Play Football")
        self.font = pygame.font.SysFont("Helvetica", 20, bold=True)

    def _xy(self, x: float, y: float) -> tuple[int, int]:
        # y grows upward in the simulation, downward on screen
        sx = self.margin + x * self.scale
        sy = self.hud_h + self.margin + (self.cfg.field_h - y) * self.scale
        return int(round(sx)), int(round(sy))

    def draw(self, core, idx: int = 0, hud: str = "") -> None:
        pg, c, s = self.pg, self.cfg, self.scale
        scr = self.screen
        scr.fill(HUD)

        fx, fy = self._xy(0, c.field_h)
        fw, fh = int(c.field_l * s), int(c.field_h * s)
        pg.draw.rect(scr, GRASS, (fx - self.margin, fy - self.margin, fw + 2 * self.margin, fh + 2 * self.margin))
        stripe = fw // 10
        for k in range(0, 10, 2):
            pg.draw.rect(scr, GRASS_DARK, (fx + k * stripe, fy, stripe, fh))

        pg.draw.rect(scr, LINE, (fx, fy, fw, fh), 2)
        mx, _ = self._xy(0.5 * c.field_l, 0)
        pg.draw.line(scr, LINE, (mx, fy), (mx, fy + fh), 2)
        pg.draw.circle(scr, LINE, self._xy(0.5 * c.field_l, 0.5 * c.field_h), int(8 * s), 2)

        gw = int(c.goal_w * s)
        gy = self._xy(0, 0.5 * (c.field_h + c.goal_w))[1]
        depth = int(2.5 * s)
        for side, color in ((0, TEAM_COLORS[0]), (1, TEAM_COLORS[1])):
            gx = fx - depth if side == 0 else fx + fw
            pg.draw.rect(scr, (220, 220, 220), (gx, gy, depth, gw))
            pg.draw.rect(scr, color, (gx, gy, depth, gw), 3)

        for i in (0, 1):
            p = core.pos[idx, i]
            center = self._xy(p[0], p[1])
            pg.draw.circle(scr, TEAM_COLORS[i], center, int(c.player_r * s))
            ready = core.cooldown[idx, i] == 0
            pg.draw.circle(scr, LINE if ready else (90, 90, 90), center, int(c.player_r * s), 2)

        b = core.ball[idx]
        pg.draw.circle(scr, BALL, self._xy(b[0], b[1]), max(3, int(c.ball_r * s)))
        pg.draw.circle(scr, HUD, self._xy(b[0], b[1]), max(3, int(c.ball_r * s)), 1)

        if hud:
            text = self.font.render(hud, True, LINE)
            scr.blit(text, (self.margin, (self.hud_h - text.get_height()) // 2))

    def flip(self) -> None:
        self.pg.display.flip()

    def rgb_array(self) -> np.ndarray:
        arr = self.pg.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).copy()

    def close(self) -> None:
        self.pg.quit()

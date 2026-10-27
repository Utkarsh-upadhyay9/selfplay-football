"""Physics and game constants.

Units are arbitrary "field units" and one simulation step. The field is
FIELD_L wide (x) and FIELD_H tall (y). Team 0 defends the goal at x = 0 and
attacks the goal at x = FIELD_L; team 1 is the mirror image.

The web demo (docs/js/game.js) re-implements this physics, so any change
here has to be mirrored there.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GameConfig:
    field_l: float = 100.0
    field_h: float = 64.0
    goal_w: float = 20.0

    player_r: float = 2.0
    player_accel: float = 0.35
    player_damping: float = 0.85
    player_max_speed: float = 2.0

    ball_r: float = 1.2
    ball_friction: float = 0.97
    ball_wall_restitution: float = 0.8
    ball_player_restitution: float = 0.3

    kick_speed: float = 4.5
    kick_reach: float = 1.5
    kick_cooldown: int = 8

    max_steps: int = 400
    kickoff_noise: float = 2.0

    @property
    def kick_range(self) -> float:
        return self.player_r + self.ball_r + self.kick_reach

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = GameConfig()

N_MOVES = 9
N_ACTIONS = N_MOVES * 2
OBS_DIM = 16

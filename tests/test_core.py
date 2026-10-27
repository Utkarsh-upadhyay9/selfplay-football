import numpy as np

from football.bots import ChaseBot, RandomBot
from football.config import DEFAULT_CONFIG, N_ACTIONS, N_MOVES, OBS_DIM
from football.core import FootballCore
from football.evaluate import play_matches


def test_shapes_and_reset():
    env = FootballCore(8, seed=0)
    obs = env.reset()
    assert obs.shape == (8, 2, OBS_DIM)
    assert obs.dtype == np.float32
    obs, sparse, shaping, done, info = env.step(np.zeros((8, 2), dtype=np.int64))
    assert sparse.shape == shaping.shape == (8, 2)
    assert done.shape == info["goal"].shape == (8,)


def test_observations_are_mirrored():
    env = FootballCore(1, seed=0)
    env.pos[0] = [[20.0, 40.0], [80.0, 40.0]]
    env.ball[0] = [50.0, 32.0]
    obs = env.observe()[0]
    # Each player sees itself at the same mirrored spot and the ball dead ahead.
    np.testing.assert_allclose(obs[0, 0:2], obs[1, 0:2], atol=1e-6)
    np.testing.assert_allclose(obs[0, 8:10], obs[1, 8:10], atol=1e-6)
    assert obs[0, 12] > 0 and obs[1, 12] > 0


def test_forward_moves_towards_opponent_goal():
    env = FootballCore(1, seed=0)
    start = env.pos[0].copy()
    for _ in range(10):
        env.step(np.array([[1, 1]]))
    assert env.pos[0, 0, 0] > start[0, 0]
    assert env.pos[0, 1, 0] < start[1, 0]


def test_players_stay_on_field():
    env = FootballCore(32, seed=1)
    rng = np.random.default_rng(0)
    c = DEFAULT_CONFIG
    for _ in range(500):
        env.step(rng.integers(0, N_ACTIONS, (32, 2)))
        assert (env.pos[..., 0] >= c.player_r - 1e-9).all()
        assert (env.pos[..., 0] <= c.field_l - c.player_r + 1e-9).all()
        assert (env.pos[..., 1] >= c.player_r - 1e-9).all()
        assert (env.pos[..., 1] <= c.field_h - c.player_r + 1e-9).all()


def test_kick_into_empty_goal_scores():
    env = FootballCore(1, seed=0, auto_reset=False)
    c = DEFAULT_CONFIG
    env.pos[0] = [[80.0, 32.0], [10.0, 5.0]]
    env.ball[0] = [80.0 + c.player_r + c.ball_r + 0.5, 32.0]
    kick_forward = 1 + N_MOVES
    goal = 0
    for _ in range(60):
        _, sparse, _, done, info = env.step(np.array([[kick_forward, 0]]))
        if done[0]:
            goal = info["goal"][0]
            assert sparse[0, 0] == 1 and sparse[0, 1] == -1
            break
    assert goal == 1


def test_episode_truncates():
    env = FootballCore(4, seed=0, auto_reset=False)
    for t in range(DEFAULT_CONFIG.max_steps):
        _, _, _, done, info = env.step(np.zeros((4, 2), dtype=np.int64))
    assert done.all() and info["truncated"].all()


def test_shaping_is_potential_difference():
    env = FootballCore(4, seed=0, auto_reset=False)
    phi0 = env.potential()
    _, _, shaping, _, _ = env.step(np.full((4, 2), 1))
    np.testing.assert_allclose(shaping, env.potential() - phi0)


def test_sides_are_balanced():
    r = play_matches(ChaseBot(), ChaseBot(), n_games=2000, seed=0)
    assert abs(r.wins - r.losses) / r.games < 0.05


def test_chase_beats_random():
    r = play_matches(ChaseBot(), RandomBot(seed=0), n_games=500, seed=0)
    assert r.win_rate > 0.75

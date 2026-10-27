import numpy as np
import torch

from football.evaluate import fit_elo
from football.policy import ActorCritic, PolicyPlayer, load_checkpoint, save_checkpoint
from football.ppo import compute_gae


def test_gae_matches_discounted_return_when_lambda_is_one():
    rewards = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, np.zeros(1, dtype=np.float32), gamma=0.9, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [0.81, 0.9, 1.0], rtol=1e-6)


def test_gae_does_not_bootstrap_across_episodes():
    rewards = np.zeros((2, 1), dtype=np.float32)
    values = np.array([[0.0], [5.0]], dtype=np.float32)
    dones = np.array([[1.0], [0.0]], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, dones, np.zeros(1, dtype=np.float32), gamma=0.99, lam=0.95)
    assert adv[0, 0] == 0.0


def test_checkpoint_roundtrip(tmp_path):
    net = ActorCritic(hidden=32)
    save_checkpoint(tmp_path / "a.pt", net, update=3)
    net2, meta = load_checkpoint(tmp_path / "a.pt")
    assert meta["update"] == 3
    x = torch.randn(5, net.obs_dim)
    torch.testing.assert_close(net.actor(x), net2.actor(x))


def test_policy_player_outputs_valid_actions():
    player = PolicyPlayer(ActorCritic(hidden=16))
    a = player(np.random.randn(7, 16).astype(np.float32))
    assert a.shape == (7,) and (a >= 0).all() and (a < 18).all()


def test_elo_orders_players():
    score = np.array([[0.5, 0.2, 0.1], [0.8, 0.5, 0.3], [0.9, 0.7, 0.5]])
    r = fit_elo(score)
    assert r[0] == 1000.0 and r[0] < r[1] < r[2]

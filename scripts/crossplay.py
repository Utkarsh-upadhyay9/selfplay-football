"""Round-robin cross-play between checkpoints of one run, plus an Elo curve.

Example:
    python scripts/crossplay.py runs/main --n 12 --games 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.bots import ChaseBot, RandomBot  # noqa: E402
from football.evaluate import crossplay, fit_elo  # noqa: E402
from football.policy import load_checkpoint, PolicyPlayer  # noqa: E402


def pick_checkpoints(ckpt_dir: Path, n: int) -> list[Path]:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if len(ckpts) <= n:
        return ckpts
    idx = np.unique(np.linspace(0, len(ckpts) - 1, n).round().astype(int))
    return [ckpts[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=12, help="number of checkpoints")
    ap.add_argument("--games", type=int, default=200, help="games per pairing")
    ap.add_argument("--out", default=None, help="output dir (default results/<run>)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out or Path("results") / run.name)
    out.mkdir(parents=True, exist_ok=True)

    paths = pick_checkpoints(run / "checkpoints", args.n)
    players, labels, steps = [RandomBot(seed=0), ChaseBot()], ["random", "chase"], [None, None]
    for p in paths:
        net, meta = load_checkpoint(p)
        players.append(PolicyPlayer(net, name=p.stem))
        labels.append(f"{meta['env_steps'] / 1e6:.1f}M")
        steps.append(meta["env_steps"])

    score, win = crossplay(players, n_games=args.games, verbose=True)
    elo = fit_elo(score, anchor=0, anchor_rating=0.0)

    np.savez(out / "crossplay.npz", score=score, win=win, labels=np.array(labels), elo=elo)
    with (out / "elo.json").open("w") as f:
        json.dump({lab: round(float(e), 1) for lab, e in zip(labels, elo)}, f, indent=2)

    k = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * k, 0.6 + 0.5 * k))
    im = ax.imshow(score, cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(k), labels, rotation=60, ha="right")
    ax.set_yticks(range(k), labels)
    for i in range(k):
        for j in range(k):
            if i != j:
                ax.text(j, i, f"{score[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(score[i, j] - 0.5) > 0.3 else "black")
    ax.set_xlabel("opponent")
    ax.set_ylabel("player")
    ax.set_title(f"Cross-play score (win=1, draw=0.5), {run.name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "crossplay_heatmap.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    xs = np.array(steps[2:]) / 1e6
    ax.plot(xs, elo[2:], "o-", label="self-play checkpoints")
    ax.axhline(elo[0], color="gray", ls="--", label="random bot")
    ax.axhline(elo[1], color="tab:orange", ls="--", label="chase bot")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("Elo (random = 0)")
    ax.set_title(f"Elo over training, {run.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "elo.png", dpi=150)
    print(f"wrote {out}/crossplay_heatmap.png and elo.png")
    for lab, e in zip(labels, elo):
        print(f"  {lab:>8s}  {e:7.1f}")


if __name__ == "__main__":
    main()

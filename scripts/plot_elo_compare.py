"""Overlay the Elo curves produced by crossplay.py for several runs.

Each run's ratings are fitted separately but anchored to the same random bot
(Elo 0), so the curves are roughly comparable.

Example:
    python scripts/plot_elo_compare.py results/main results/main_seed1 results/sparse results/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/elo_compare.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4))
    chase = []
    for d in args.dirs:
        data = np.load(Path(d) / "crossplay.npz")
        labels, elo = list(data["labels"]), data["elo"]
        steps = [float(lab.rstrip("M")) for lab in labels[2:]]
        ax.plot(steps, elo[2:], "o-", ms=4, label=Path(d).name)
        chase.append(elo[1])
    ax.axhline(0, color="gray", ls="--", lw=1, label="random bot")
    ax.axhline(float(np.mean(chase)), color="black", ls=":", lw=1, label="chase bot (mean)")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("Elo (random bot = 0)")
    ax.set_title("Elo of checkpoints over training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

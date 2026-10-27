"""Plot learning curves from one or more runs' metrics.csv.

Example:
    python scripts/plot_curves.py runs/main runs/sparse runs/latest
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        vals = [float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]
        cols[k] = np.array(vals)
    return cols


def smooth(y: np.ndarray, k: int = 20) -> np.ndarray:
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1) : i + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/learning_curves.png")
    args = ap.parse_args()

    panels = [
        ("win_vs_chase", "win rate vs chase bot", True),
        ("win_vs_random", "win rate vs random bot", True),
        ("train_draw", "draw rate in self-play", False),
        ("entropy", "policy entropy", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for run in args.runs:
        m = read(Path(run) / "metrics.csv")
        x = m["env_steps"] / 1e6
        for ax, (key, title, sparse_eval) in zip(axes.flat, panels):
            y = m[key]
            if sparse_eval:
                ok = ~np.isnan(y)
                ax.plot(x[ok], smooth(y[ok], 4), label=Path(run).name)
            else:
                ax.plot(x, smooth(y), label=Path(run).name)
            ax.set_title(title)
            ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("environment steps (millions)")
    axes[0, 0].legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

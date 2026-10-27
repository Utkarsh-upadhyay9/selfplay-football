"""Baselines and ablations: final agents of several runs vs bots and each other.

Example:
    python scripts/ablations.py runs/main runs/sparse runs/latest --games 1000
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
from football.evaluate import crossplay, play_matches  # noqa: E402
from football.policy import load_player  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = [Path(r).name for r in args.runs]
    players = [load_player(Path(r) / "final.pt", name=n) for r, n in zip(args.runs, names)]

    rows = []
    for name, p in zip(names, players):
        rr = play_matches(p, RandomBot(seed=1), args.games, seed=11)
        rc = play_matches(p, ChaseBot(), args.games, seed=12)
        rows.append({
            "run": name,
            "vs_random": {"win": rr.win_rate, "draw": rr.draws / rr.games, "loss": rr.losses / rr.games},
            "vs_chase": {"win": rc.win_rate, "draw": rc.draws / rc.games, "loss": rc.losses / rc.games},
        })
        print(f"{name:>12s}  vs random W/D/L {rr.wins}/{rr.draws}/{rr.losses}   vs chase W/D/L {rc.wins}/{rc.draws}/{rc.losses}")

    score, win = crossplay(players, n_games=args.games, seed=99)
    print("\nhead-to-head score (row vs column):")
    print(" " * 14 + "".join(f"{n:>12s}" for n in names))
    for n, row in zip(names, score):
        print(f"{n:>12s}  " + "".join(f"{v:12.2f}" for v in row))

    with (out / "ablations.json").open("w") as f:
        json.dump({"games": args.games, "baselines": rows, "names": names, "head_to_head_score": score.tolist(),
                   "head_to_head_win": np.nan_to_num(win).tolist()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(names))
    w = 0.38
    for k, (key, color) in enumerate((("vs_random", "tab:gray"), ("vs_chase", "tab:orange"))):
        wins = [r[key]["win"] for r in rows]
        draws = [r[key]["draw"] for r in rows]
        ax.bar(x + (k - 0.5) * w, wins, w, color=color, label=f"win {key.replace('_', ' ')}")
        ax.bar(x + (k - 0.5) * w, draws, w, bottom=wins, color=color, alpha=0.35, label=f"draw {key.replace('_', ' ')}")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of games")
    ax.set_title(f"Final agents vs scripted baselines ({args.games} games each)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "baselines.png", dpi=150)
    print(f"\nwrote {out}/ablations.json and baselines.png")


if __name__ == "__main__":
    main()

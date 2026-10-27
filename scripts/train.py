"""Train a 1v1 agent with self-play PPO.

Examples:
    python scripts/train.py --name main
    python scripts/train.py --name sparse --sparse
    python scripts/train.py --name latest --opponents latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.ppo import PPOConfig  # noqa: E402
from football.selfplay import SelfPlayTrainer, TrainConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main")
    ap.add_argument("--total-steps", type=float, default=30e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--opponents", choices=["pool", "latest"], default="pool")
    ap.add_argument("--sparse", action="store_true", help="disable reward shaping")
    ap.add_argument("--shaping-end-frac", type=float, default=0.5)
    ap.add_argument("--draw-penalty", type=float, default=0.0, help="reward subtracted from both sides on a timeout")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    cfg = TrainConfig(
        name=args.name,
        total_steps=int(args.total_steps),
        seed=args.seed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        opponent_mode=args.opponents,
        shaping=not args.sparse,
        shaping_end_frac=args.shaping_end_frac,
        draw_penalty=args.draw_penalty,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        torch_threads=args.threads,
        out_dir=args.out_dir,
        ppo=PPOConfig(lr=args.lr, ent_coef=args.ent_coef),
    )
    final = SelfPlayTrainer(cfg).train()
    print(f"saved {final}")


if __name__ == "__main__":
    main()

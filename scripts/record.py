"""Record matches between two players to an animated GIF.

Example:
    python scripts/record.py models/final.pt chase --games 3 --out media/final_vs_chase.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.config import DEFAULT_CONFIG  # noqa: E402
from football.core import FootballCore  # noqa: E402
from football.players import make_player  # noqa: E402
from football.render import Renderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", help="red player (checkpoint, chase or random)")
    ap.add_argument("right", help="blue player")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="keep every k-th frame")
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--out", default="media/match.gif")
    args = ap.parse_args()

    left, right = make_player(args.left, seed=args.seed), make_player(args.right, seed=args.seed + 1)
    names = (Path(args.left).stem, Path(args.right).stem)
    core = FootballCore(1, DEFAULT_CONFIG, seed=args.seed, auto_reset=False)
    renderer = Renderer(DEFAULT_CONFIG, scale=args.scale, headless=True, hud_h=28)

    frames = []
    score = [0, 0]
    for game in range(args.games):
        obs = core.reset()
        t = 0
        while True:
            if t % args.every == 0:
                renderer.draw(core, 0, f"{names[0]} {score[0]} - {score[1]} {names[1]}   game {game + 1}/{args.games}")
                frames.append(Image.fromarray(renderer.rgb_array()))
            acts = np.array([[left(obs[:, 0])[0], right(obs[:, 1])[0]]])
            obs, _, _, done, info = core.step(acts)
            t += 1
            if done[0]:
                g = int(info["goal"][0])
                if g == 1:
                    score[0] += 1
                elif g == -1:
                    score[1] += 1
                renderer.draw(core, 0, f"{names[0]} {score[0]} - {score[1]} {names[1]}   game {game + 1}/{args.games}")
                frames.extend([Image.fromarray(renderer.rgb_array())] * 10)
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / 30 * args.every), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB), final score {score}")


if __name__ == "__main__":
    main()

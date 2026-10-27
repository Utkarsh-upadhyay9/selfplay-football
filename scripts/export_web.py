"""Export actor weights to JSON for the browser demo in docs/.

Example:
    python scripts/export_web.py rookie=models/early.pt pro=models/final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football.policy import load_checkpoint  # noqa: E402


def export(path: str | Path, name: str, out: Path) -> Path:
    net, meta = load_checkpoint(path)
    layers = []
    for module in net.actor:
        if hasattr(module, "weight"):
            layers.append(
                {
                    "W": [[round(float(w), 5) for w in row] for row in module.weight.detach().numpy()],
                    "b": [round(float(b), 5) for b in module.bias.detach().numpy()],
                }
            )
    spec = {"name": name, "arch": net.arch(), "env_steps": meta.get("env_steps"), "layers": layers}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, separators=(",", ":")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="name=checkpoint.pt pairs")
    ap.add_argument("--out-dir", default="docs/models")
    args = ap.parse_args()
    for item in args.models:
        name, path = item.split("=", 1)
        out = export(path, name, Path(args.out_dir) / f"{name}.json")
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

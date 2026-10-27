"""Plot the CartPole PPO sanity check produced by sanity_cartpole.py."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = list(csv.DictReader(Path("runs/sanity_cartpole/metrics.csv").open()))
x = [int(r["env_steps"]) / 1e3 for r in rows]
y = [float(r["mean_return"]) for r in rows]

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(x, y)
ax.axhline(500, color="gray", ls="--", lw=1, label="max return")
ax.set_xlabel("environment steps (thousands)")
ax.set_ylabel("mean return (last 20 episodes)")
ax.set_title("PPO sanity check on CartPole-v1")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/cartpole_sanity.png", dpi=150)
print("wrote results/cartpole_sanity.png")

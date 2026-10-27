# Self-Play Deep RL for 2D Football

CSE 4633 Artificial Intelligence midterm project · Utkarsh Upadhyay (solo)

**Play it in the browser:** <https://utkarsh-upadhyay9.github.io/selfplay-football/>
· **Report:** [report/REPORT.md](report/REPORT.md)

![Final agent (red) vs the scripted chase bot](media/pro_vs_chase.gif)

This project trains agents to play a small top-down 1v1 football game with
PPO and self-play, from scratch, on a laptop CPU, and lets you play against
them. The agent sees a 16-number state vector (positions and velocities from
its own point of view) and picks one of 18 actions (move direction × kick).

## Results at a glance

- With an **opponent pool** of past checkpoints, Elo rises steadily over 30M
  steps, to about +300 over a random player. Later checkpoints beat earlier
  ones in 83–89% of pairings.
- **Latest-only self-play collapses.** It peaks at 5.7M steps, then the two
  copies settle into a 100%-draw equilibrium, and the final agent loses 79%
  of games to the scripted bot.
- **Reward shaping** mainly speeds up the first few million steps. The
  sparse-reward run catches up by about 8M steps.
- The final agent wins **80%** of games against a "chase the ball and shoot"
  bot and **70%** against a random player (where most of the remaining games
  are draws).

![Elo over training](results/elo_compare.png)

## Layout

```
football/
  core.py        vectorized numpy physics for N games at once
  env.py         Gymnasium single-agent wrapper
  render.py      pygame renderer
  bots.py        random and chase-the-ball scripted opponents
  policy.py      actor-critic MLP, checkpoints
  ppo.py         PPO + GAE
  selfplay.py    self-play trainer (opponent pool / latest-only)
  evaluate.py    matches, cross-play round robin, Elo fit
scripts/
  train.py             train a run
  sanity_cartpole.py   PPO sanity check on CartPole-v1
  crossplay.py         cross-play heatmap + Elo curve for one run
  ablations.py         final agents vs bots and each other
  plot_curves.py       learning curves
  play.py              play against an agent with the keyboard (pygame)
  record.py            record matches to GIF
  export_web.py        export policies to JSON for the browser
docs/            the GitHub Pages site (JS port of the game + policy inference)
models/          selected checkpoints (final.pt is the final agent)
runs/            per-run config, metrics.csv and final checkpoint
results/         figures and evaluation outputs
report/          the write-up
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/play.py --agent models/final.pt            # play (arrows/WASD + Space)
python scripts/play.py --agent models/final.pt --fps 15   # slower, easier
python scripts/play.py --watch --agent models/final.pt --opponent chase

python scripts/train.py --name my_run                     # ~16 min for 30M steps on an M5
python -m pytest -q
```

See the report for the full set of commands used to produce every figure.

// Forward pass of the exported actor network (Linear -> tanh -> Linear -> tanh -> Linear).

import { CFG, MOVE_DIRS, N_ACTIONS, N_MOVES } from "./game.js";

export class Policy {
  constructor(spec) {
    this.name = spec.name;
    this.layers = spec.layers.map((l) => ({
      W: l.W.map((row) => Float32Array.from(row)),
      b: Float32Array.from(l.b),
    }));
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to load ${url}`);
    return new Policy(await res.json());
  }

  logits(obs) {
    let x = obs;
    this.layers.forEach((layer, k) => {
      const out = new Float32Array(layer.b.length);
      for (let i = 0; i < out.length; i++) {
        const w = layer.W[i];
        let s = layer.b[i];
        for (let j = 0; j < x.length; j++) s += w[j] * x[j];
        out[i] = k < this.layers.length - 1 ? Math.tanh(s) : s;
      }
      x = out;
    });
    return x;
  }

  act(obs, greedy = false, rng = Math.random) {
    const z = this.logits(obs);
    let best = 0;
    for (let i = 1; i < z.length; i++) if (z[i] > z[best]) best = i;
    if (greedy) return best;
    let total = 0;
    const p = new Float64Array(z.length);
    for (let i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - z[best]); total += p[i]; }
    let u = rng() * total;
    for (let i = 0; i < p.length; i++) { u -= p[i]; if (u <= 0) return i; }
    return p.length - 1;
  }
}

// Port of football/bots.py ChaseBot.
export class ChaseBot {
  constructor(cfg = CFG, align = 0.6) {
    this.name = "chase";
    this.cfg = cfg;
    this.align = align;
  }

  act(obs) {
    const c = this.cfg;
    const hl = 0.5 * c.field_l, hh = 0.5 * c.field_h;
    const me = [obs[0] * hl, obs[1] * hh];
    const ball = [obs[8] * hl, obs[9] * hh];
    let ax = hl - ball[0], ay = -ball[1];
    const an = Math.max(Math.hypot(ax, ay), 1e-8);
    ax /= an; ay /= an;
    let target = [ball[0] - ax * (c.player_r + c.ball_r + 0.5), ball[1] - ay * (c.player_r + c.ball_r + 0.5)];

    const px = -ay, py = ax;
    const ox = me[0] - ball[0], oy = me[1] - ball[1];
    if (ox * ax + oy * ay > -1.0) {
      const side = ox * px + oy * py >= 0 ? 1 : -1;
      const off = 2 * c.player_r + c.ball_r + 2.0;
      target = [ball[0] - ax * 2 + px * side * off, ball[1] - ay * 2 + py * side * off];
    }
    const tx = target[0] - me[0], ty = target[1] - me[1];
    const distTarget = Math.hypot(tx, ty);
    const bx = ball[0] - me[0], by = ball[1] - me[1];
    const distBall = Math.hypot(bx, by);
    const dm = Math.max(distBall, 1e-8);
    const linedUp = (bx / dm) * ax + (by / dm) * ay > this.align;
    const [hx, hy] = linedUp && distBall < 3 * c.kick_range ? [bx, by] : [tx, ty];

    let move = 1, best = -Infinity;
    for (let m = 1; m < N_MOVES; m++) {
      const s = MOVE_DIRS[m][0] * hx + MOVE_DIRS[m][1] * hy;
      if (s > best) { best = s; move = m; }
    }
    if (distTarget < 0.5 && !linedUp) move = 0;
    const kick = distBall <= c.kick_range && linedUp;
    return move + (kick ? N_MOVES : 0);
  }
}

export class RandomBot {
  constructor() { this.name = "random"; }
  act() { return Math.floor(Math.random() * N_ACTIONS); }
}

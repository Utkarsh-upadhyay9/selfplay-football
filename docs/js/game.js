// JavaScript port of football/core.py for a single game.
// Must stay in sync with the Python physics (tests/test_web_parity.py checks it).

export const CFG = {
  field_l: 100.0,
  field_h: 64.0,
  goal_w: 20.0,
  player_r: 2.0,
  player_accel: 0.35,
  player_damping: 0.85,
  player_max_speed: 2.0,
  ball_r: 1.2,
  ball_friction: 0.97,
  ball_wall_restitution: 0.8,
  ball_player_restitution: 0.3,
  kick_speed: 4.5,
  kick_reach: 1.5,
  kick_cooldown: 8,
  max_steps: 400,
  kickoff_noise: 2.0,
};
CFG.kick_range = CFG.player_r + CFG.ball_r + CFG.kick_reach;

export const N_MOVES = 9;
export const N_ACTIONS = 18;
export const OBS_DIM = 16;

const S = Math.SQRT1_2;
export const MOVE_DIRS = [
  [0, 0], [1, 0], [S, S], [0, 1], [-S, S], [-1, 0], [-S, -S], [0, -1], [S, -S],
];

const uniform = (rng) => rng() * 2 - 1;

export class Game {
  constructor(cfg = CFG, rng = Math.random) {
    this.cfg = cfg;
    this.rng = rng;
    this.reset();
  }

  reset() {
    const c = this.cfg;
    const n = c.kickoff_noise;
    const r = this.rng;
    this.pos = [
      [0.25 * c.field_l + uniform(r) * n, 0.5 * c.field_h + uniform(r) * n],
      [0.75 * c.field_l + uniform(r) * n, 0.5 * c.field_h + uniform(r) * n],
    ];
    this.vel = [[0, 0], [0, 0]];
    this.ball = [0.5 * c.field_l + uniform(r) * 0.5, 0.5 * c.field_h + uniform(r) * 0.5];
    this.ballVel = [0, 0];
    this.cooldown = [0, 0];
    this.t = 0;
  }

  // actions: [a0, a1] in 0..17. Returns {goal, done, truncated}; goal is +1 when team 0 scores.
  step(actions) {
    const c = this.cfg;
    this._movePlayers(actions.map((a) => a % N_MOVES));
    this._collidePlayers();
    this._ballContact();
    this._kick(actions.map((a) => a >= N_MOVES));
    const goal = this._moveBall();
    this.t += 1;
    const truncated = this.t >= c.max_steps && goal === 0;
    return { goal, done: goal !== 0 || truncated, truncated };
  }

  _movePlayers(moves) {
    const c = this.cfg;
    const r = c.player_r;
    const hi = [c.field_l - r, c.field_h - r];
    for (let i = 0; i < 2; i++) {
      const d = MOVE_DIRS[moves[i]];
      const dx = i === 1 ? -d[0] : d[0];
      const v = this.vel[i];
      v[0] = v[0] * c.player_damping + dx * c.player_accel;
      v[1] = v[1] * c.player_damping + d[1] * c.player_accel;
      const speed = Math.hypot(v[0], v[1]);
      const k = Math.min(1, c.player_max_speed / Math.max(speed, 1e-8));
      v[0] *= k;
      v[1] *= k;
      const p = this.pos[i];
      p[0] += v[0];
      p[1] += v[1];
      for (let ax = 0; ax < 2; ax++) {
        if (p[ax] < r || p[ax] > hi[ax]) {
          p[ax] = Math.min(Math.max(p[ax], r), hi[ax]);
          v[ax] = 0;
        }
      }
    }
  }

  _collidePlayers() {
    const r = this.cfg.player_r;
    const [a, b] = this.pos;
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dist = Math.hypot(dx, dy);
    const overlap = 2 * r - dist;
    if (overlap <= 0) return;
    let nx = 1, ny = 0;
    if (dist > 1e-8) { nx = dx / dist; ny = dy / dist; }
    const push = 0.5 * overlap;
    a[0] -= nx * push; a[1] -= ny * push;
    b[0] += nx * push; b[1] += ny * push;
  }

  _ballContact() {
    const c = this.cfg;
    const minD = c.player_r + c.ball_r;
    const e = c.ball_player_restitution;
    const first = this.t % 2;
    for (const who of [first, 1 - first]) {
      const p = this.pos[who];
      const v = this.vel[who];
      const rx = this.ball[0] - p[0];
      const ry = this.ball[1] - p[1];
      const dist = Math.hypot(rx, ry);
      if (dist >= minD) continue;
      let nx = 1, ny = 0;
      if (dist >= 1e-8) { nx = rx / dist; ny = ry / dist; }
      this.ball[0] = p[0] + nx * minD;
      this.ball[1] = p[1] + ny * minD;
      const vn = (this.ballVel[0] - v[0]) * nx + (this.ballVel[1] - v[1]) * ny;
      if (vn < 0) {
        this.ballVel[0] -= (1 + e) * vn * nx;
        this.ballVel[1] -= (1 + e) * vn * ny;
      }
    }
  }

  _kick(kicks) {
    const c = this.cfg;
    let ix = 0, iy = 0, kicked = false;
    const can = [false, false];
    for (let i = 0; i < 2; i++) {
      this.cooldown[i] = Math.max(this.cooldown[i] - 1, 0);
      const rx = this.ball[0] - this.pos[i][0];
      const ry = this.ball[1] - this.pos[i][1];
      const dist = Math.hypot(rx, ry);
      can[i] = kicks[i] && this.cooldown[i] === 0 && dist <= c.kick_range;
      if (can[i]) {
        const m = Math.max(dist, 1e-8);
        ix += (rx / m) * c.kick_speed;
        iy += (ry / m) * c.kick_speed;
        kicked = true;
      }
    }
    if (!kicked) return;
    this.ballVel = [ix, iy];
    for (let i = 0; i < 2; i++) if (can[i]) this.cooldown[i] = c.kick_cooldown;
  }

  _moveBall() {
    const c = this.cfg;
    const b = this.ball;
    const bv = this.ballVel;
    b[0] += bv[0];
    b[1] += bv[1];
    bv[0] *= c.ball_friction;
    bv[1] *= c.ball_friction;

    const r = c.ball_r;
    const rest = c.ball_wall_restitution;
    if (b[1] < r) { b[1] = 2 * r - b[1]; bv[1] *= -rest; }
    else if (b[1] > c.field_h - r) { b[1] = 2 * (c.field_h - r) - b[1]; bv[1] *= -rest; }

    const inMouth = Math.abs(b[1] - 0.5 * c.field_h) < 0.5 * c.goal_w;
    if (!inMouth) {
      if (b[0] < r) { b[0] = 2 * r - b[0]; bv[0] *= -rest; }
      else if (b[0] > c.field_l - r) { b[0] = 2 * (c.field_l - r) - b[0]; bv[0] *= -rest; }
    }
    if (inMouth && b[0] > c.field_l) return 1;
    if (inMouth && b[0] < 0) return -1;
    return 0;
  }

  // Ego-centric observation for player i (team 1 is mirrored along x).
  observe(i) {
    const c = this.cfg;
    const hl = 0.5 * c.field_l;
    const hh = 0.5 * c.field_h;
    const j = 1 - i;
    const sx = i === 0 ? 1 : -1;
    const P = (p) => [((p[0] - hl) / hl) * sx, (p[1] - hh) / hh];
    const me = P(this.pos[i]);
    const op = P(this.pos[j]);
    const ball = P(this.ball);
    const o = new Float32Array(OBS_DIM);
    o[0] = me[0]; o[1] = me[1];
    o[2] = (this.vel[i][0] / c.player_max_speed) * sx; o[3] = this.vel[i][1] / c.player_max_speed;
    o[4] = op[0]; o[5] = op[1];
    o[6] = (this.vel[j][0] / c.player_max_speed) * sx; o[7] = this.vel[j][1] / c.player_max_speed;
    o[8] = ball[0]; o[9] = ball[1];
    o[10] = (this.ballVel[0] / c.kick_speed) * sx; o[11] = this.ballVel[1] / c.kick_speed;
    o[12] = ball[0] - me[0]; o[13] = ball[1] - me[1];
    o[14] = this.cooldown[i] / c.kick_cooldown;
    o[15] = this.t / c.max_steps;
    return o;
  }

  // Turn screen-space input (dx right, dy up) into an action for player `side`.
  static keyboardAction(dx, dy, kick, side) {
    if (side === 1) dx = -dx;
    let move = 0;
    if (dx || dy) {
      let best = -Infinity;
      for (let m = 1; m < N_MOVES; m++) {
        const s = MOVE_DIRS[m][0] * dx + MOVE_DIRS[m][1] * dy;
        if (s > best) { best = s; move = m; }
      }
    }
    return move + (kick ? N_MOVES : 0);
  }
}

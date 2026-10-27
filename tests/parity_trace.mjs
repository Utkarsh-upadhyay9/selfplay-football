// Replays a fixed state + action sequence through docs/js/game.js and prints
// the trajectory as JSON. Used by tests/test_web_parity.py.
import { readFileSync } from "node:fs";
import { Game } from "../docs/js/game.js";
import { ChaseBot } from "../docs/js/policy.js";

const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
const game = new Game();
game.pos = input.pos;
game.vel = input.vel;
game.ball = input.ball;
game.ballVel = input.ball_vel;
game.cooldown = input.cooldown;
game.t = 0;

const bot = new ChaseBot();
const trace = [];
for (const a of input.actions) {
  const chase = bot.act(game.observe(0));
  const r = game.step(a);
  trace.push({
    pos: game.pos.map((p) => [...p]),
    ball: [...game.ball],
    goal: r.goal,
    obs: Array.from(game.observe(1)),
    chase,
  });
  if (r.done) break;
}
console.log(JSON.stringify(trace));

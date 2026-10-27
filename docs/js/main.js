import { CFG, Game } from "./game.js";
import { ChaseBot, Policy, RandomBot } from "./policy.js";

const canvas = document.getElementById("pitch");
const ctx = canvas.getContext("2d");
const el = (id) => document.getElementById(id);

const SCALE = 8;
const MARGIN = 24;
canvas.width = CFG.field_l * SCALE + 2 * MARGIN;
canvas.height = CFG.field_h * SCALE + 2 * MARGIN;

const COLORS = {
  grass: "#2e7d32",
  grassDark: "#28702d",
  line: "#ebf5eb",
  team: ["#e53935", "#1e88e5"],
  ball: "#fafafa",
};

const state = {
  game: new Game(),
  opponents: {},
  agent: null,
  left: null, // null = human
  score: [0, 0],
  firstTo: 5,
  fps: 30,
  paused: true,
  over: false,
  watch: false,
  greedy: false,
  acc: 0,
  last: performance.now(),
  flash: 0,
};

const keys = new Set();
window.addEventListener("keydown", (e) => {
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(e.key)) e.preventDefault();
  keys.add(e.key.toLowerCase());
  if (e.key === "p" || e.key === "P") togglePause();
  if (e.key === "r" || e.key === "R") restart();
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => keys.clear());

// Minimal on-screen controls for touch devices.
const touch = { dx: 0, dy: 0, kick: false };

function humanAction() {
  const has = (...k) => k.some((x) => keys.has(x));
  let dx = (has("arrowright", "d") ? 1 : 0) - (has("arrowleft", "a") ? 1 : 0);
  let dy = (has("arrowup", "w") ? 1 : 0) - (has("arrowdown", "s") ? 1 : 0);
  if (!dx && !dy) { dx = touch.dx; dy = touch.dy; }
  return Game.keyboardAction(dx, dy, has(" ") || touch.kick, 0);
}
document.querySelectorAll("[data-dir]").forEach((b) => {
  const [dx, dy] = b.dataset.dir.split(",").map(Number);
  const on = (e) => { e.preventDefault(); touch.dx = dx; touch.dy = dy; };
  const off = (e) => { e.preventDefault(); touch.dx = 0; touch.dy = 0; };
  b.addEventListener("pointerdown", on);
  b.addEventListener("pointerup", off);
  b.addEventListener("pointerleave", off);
});
const kickBtn = el("touch-kick");
if (kickBtn) {
  kickBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); touch.kick = true; });
  kickBtn.addEventListener("pointerup", () => { touch.kick = false; });
  kickBtn.addEventListener("pointerleave", () => { touch.kick = false; });
}

function playerAct(player, side) {
  const obs = state.game.observe(side);
  return player.act(obs, state.greedy);
}

function tick() {
  const g = state.game;
  const a0 = state.left ? playerAct(state.left, 0) : humanAction();
  const a1 = playerAct(state.agent, 1);
  const r = g.step([a0, a1]);
  if (r.done) {
    if (r.goal === 1) state.score[0] += 1;
    if (r.goal === -1) state.score[1] += 1;
    if (r.goal !== 0) state.flash = 30;
    g.reset();
    if (Math.max(...state.score) >= state.firstTo) finishMatch();
    updateScore();
  }
}

function finishMatch() {
  state.over = true;
  state.paused = true;
  const [you, ai] = state.score;
  if (!state.watch) recordResult(you > ai);
  const who = state.watch ? (you > ai ? "Left player wins" : "Agent wins") : you > ai ? "You win!" : "The agent wins";
  el("status").textContent = `${who} ${you}–${ai}. Press Restart (or R) for a rematch.`;
  el("btn-start").textContent = "Restart";
}

function recordResult(won) {
  const key = `record:${el("opponent").value}`;
  const rec = JSON.parse(localStorage.getItem(key) || '{"w":0,"l":0}');
  if (won) rec.w += 1; else rec.l += 1;
  localStorage.setItem(key, JSON.stringify(rec));
  updateRecord();
}

function updateRecord() {
  const key = `record:${el("opponent").value}`;
  const rec = JSON.parse(localStorage.getItem(key) || '{"w":0,"l":0}');
  el("record").textContent = `Your record vs this agent: ${rec.w} W / ${rec.l} L`;
}

function updateScore() {
  el("score-left").textContent = state.score[0];
  el("score-right").textContent = state.score[1];
  el("name-left").textContent = state.watch ? el("left-player").selectedOptions[0].text : "You";
  el("name-right").textContent = el("opponent").selectedOptions[0].text;
}

// ---------------------------------------------------------------- drawing
const X = (x) => MARGIN + x * SCALE;
const Y = (y) => MARGIN + (CFG.field_h - y) * SCALE;

function draw() {
  const c = CFG;
  const g = state.game;
  ctx.fillStyle = COLORS.grass;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const fw = c.field_l * SCALE, fh = c.field_h * SCALE;
  ctx.fillStyle = COLORS.grassDark;
  for (let k = 0; k < 10; k += 2) ctx.fillRect(X(0) + (k * fw) / 10, Y(c.field_h), fw / 10, fh);

  ctx.strokeStyle = COLORS.line;
  ctx.lineWidth = 2;
  ctx.strokeRect(X(0), Y(c.field_h), fw, fh);
  ctx.beginPath();
  ctx.moveTo(X(c.field_l / 2), Y(c.field_h));
  ctx.lineTo(X(c.field_l / 2), Y(0));
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(X(c.field_l / 2), Y(c.field_h / 2), 8 * SCALE, 0, 2 * Math.PI);
  ctx.stroke();

  const gTop = Y(c.field_h / 2 + c.goal_w / 2);
  const depth = 2.5 * SCALE;
  [[X(0) - depth, COLORS.team[0]], [X(c.field_l), COLORS.team[1]]].forEach(([gx, col]) => {
    ctx.fillStyle = "#dcdcdc";
    ctx.fillRect(gx, gTop, depth, c.goal_w * SCALE);
    ctx.strokeStyle = col;
    ctx.lineWidth = 3;
    ctx.strokeRect(gx, gTop, depth, c.goal_w * SCALE);
  });

  for (let i = 0; i < 2; i++) {
    const [px, py] = g.pos[i];
    ctx.beginPath();
    ctx.arc(X(px), Y(py), c.player_r * SCALE, 0, 2 * Math.PI);
    ctx.fillStyle = COLORS.team[i];
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = g.cooldown[i] === 0 ? COLORS.line : "#5a5a5a";
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(X(g.ball[0]), Y(g.ball[1]), Math.max(3, c.ball_r * SCALE), 0, 2 * Math.PI);
  ctx.fillStyle = COLORS.ball;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#141414";
  ctx.stroke();

  if (state.flash > 0) {
    ctx.fillStyle = `rgba(255,255,255,${(state.flash / 30) * 0.5})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#141414";
    ctx.font = "bold 64px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("GOAL!", canvas.width / 2, canvas.height / 2 + 20);
    state.flash -= 1;
  }
  if (state.paused && !state.over) {
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 36px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(state.watch ? "Press Start to watch" : "Press Start to play", canvas.width / 2, canvas.height / 2 + 12);
  }
}

function loop(now) {
  const dt = Math.min(now - state.last, 250);
  state.last = now;
  if (!state.paused && state.agent) {
    state.acc += dt;
    const stepMs = 1000 / state.fps;
    while (state.acc >= stepMs && !state.paused) {
      tick();
      state.acc -= stepMs;
    }
  }
  draw();
  requestAnimationFrame(loop);
}

// --------------------------------------------------------------- controls
function playerFor(id) {
  if (id === "chase") return new ChaseBot();
  if (id === "random") return new RandomBot();
  return state.opponents[id];
}

function restart() {
  state.score = [0, 0];
  state.over = false;
  state.game.reset();
  state.acc = 0;
  state.paused = false;
  el("btn-start").textContent = "Restart";
  el("btn-pause").textContent = "Pause";
  el("status").textContent = state.watch
    ? "Watching two players."
    : "Arrow keys / WASD to move, Space to kick. You are red, attacking right.";
  updateScore();
}

function togglePause() {
  if (state.over) return;
  state.paused = !state.paused;
  el("btn-pause").textContent = state.paused ? "Resume" : "Pause";
}

function applySettings() {
  state.agent = playerFor(el("opponent").value);
  state.watch = el("mode").value === "watch";
  state.left = state.watch ? playerFor(el("left-player").value) : null;
  el("left-player-wrap").hidden = !state.watch;
  state.firstTo = Number(el("first-to").value);
  state.greedy = el("greedy").checked;
  updateScore();
  updateRecord();
}

el("speed").addEventListener("input", (e) => {
  state.fps = Number(e.target.value);
  el("speed-val").textContent = `${state.fps} steps/s`;
});
["opponent", "mode", "left-player", "first-to", "greedy"].forEach((id) =>
  el(id).addEventListener("change", () => { applySettings(); restart(); state.paused = true; el("btn-start").textContent = "Start"; })
);
el("btn-start").addEventListener("click", () => { applySettings(); restart(); canvas.focus(); });
el("btn-pause").addEventListener("click", togglePause);

async function init() {
  const manifest = await (await fetch("models/manifest.json")).json();
  const selects = [el("opponent"), el("left-player")];
  const policies = await Promise.all(manifest.models.map((m) => Policy.load(`models/${m.id}.json`)));
  manifest.models.forEach((m, i) => {
    state.opponents[m.id] = policies[i];
    for (const s of selects) s.add(new Option(m.label, m.id));
  });
  for (const s of selects) {
    s.add(new Option("Scripted chase bot", "chase"));
    s.add(new Option("Random player", "random"));
  }
  el("opponent").value = manifest.default;
  el("left-player").value = "chase";
  applySettings();
  el("status").textContent = "Pick an opponent and press Start.";
  requestAnimationFrame(loop);
}

init().catch((err) => {
  el("status").textContent = `Could not load models: ${err.message}`;
  console.error(err);
});

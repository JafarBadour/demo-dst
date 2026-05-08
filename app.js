const TAU = Math.PI * 2;
const MAX_EPOCHS = 200;
const INITIAL_SEED = 42;

let rngState = INITIAL_SEED >>> 0;

function seededRandom() {
  rngState = (rngState + 0x6d2b79f5) >>> 0;
  let t = rngState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function resetSeed() {
  rngState = INITIAL_SEED >>> 0;
}

const ui = {
  mode: document.getElementById("modeSelect"),
  method: document.getElementById("methodSelect"),
  sparsity: document.getElementById("sparsitySlider"),
  sparsityValue: document.getElementById("sparsityValue"),
  rewireEvery: document.getElementById("rewireEveryInput"),
  speed: document.getElementById("speedSlider"),
  play: document.getElementById("playBtn"),
  pause: document.getElementById("pauseBtn"),
  step: document.getElementById("stepBtn"),
  reset: document.getElementById("resetBtn"),
  epochReadout: document.getElementById("epochReadout"),
  statsReadout: document.getElementById("statsReadout"),
  phaseBadge: document.getElementById("phaseBadge"),
  networkCanvas: document.getElementById("networkCanvas"),
  metricsCanvas: document.getElementById("metricsCanvas"),
};

const state = {
  running: false,
  autoStoppedAtPerfect: false,
  epoch: 0,
  timer: null,
  phase: "idle",
  pulseT: 0,
  lastRewireEpoch: -999,
  mode: "supervised",
  method: "rigl",
  sparsity: 0.5,
  rewireEvery: 8,
  speed: 3,
  history: [],
  rewireEpochs: [],
  model: null,
  runtime: null,
  vis: null,
};

function reachedPerfectAccuracy() {
  const last = state.history.at(-1);
  return Boolean(last && last.acc >= 0.9995);
}

function randn(scale = 1) {
  const u1 = Math.max(seededRandom(), 1e-9);
  const u2 = seededRandom();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TAU * u2) * scale;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function softmax(arr) {
  const m = Math.max(...arr);
  const ex = arr.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

function tanh(x) {
  return Math.tanh(x);
}

function dtanh(x) {
  const t = Math.tanh(x);
  return 1 - t * t;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function createLayer(inDim, outDim, density) {
  const w = [];
  const b = [];
  const m = [];
  const gradScore = [];
  const activeTarget = Math.max(1, Math.floor(inDim * outDim * density));
  const allIdx = [];
  for (let o = 0; o < outDim; o += 1) {
    w[o] = [];
    m[o] = [];
    gradScore[o] = [];
    for (let i = 0; i < inDim; i += 1) {
      w[o][i] = randn(0.2);
      m[o][i] = false;
      gradScore[o][i] = 0;
      allIdx.push([o, i]);
    }
    b[o] = 0;
  }
  for (let k = allIdx.length - 1; k > 0; k -= 1) {
    const r = Math.floor(seededRandom() * (k + 1));
    const tmp = allIdx[k];
    allIdx[k] = allIdx[r];
    allIdx[r] = tmp;
  }
  for (let k = 0; k < activeTarget; k += 1) {
    const [o, i] = allIdx[k];
    m[o][i] = true;
  }
  return { inDim, outDim, w, b, m, gradScore };
}

function createModel(mode, sparsity) {
  const density = 1 - sparsity;
  let dims;
  let outType;
  if (mode === "supervised") {
    dims = [2, 16, 1];
    outType = "binary";
  } else if (mode === "reinforcement") {
    dims = [3, 18, 3];
    outType = "policy";
  } else {
    dims = [2, 16, 2];
    outType = "regression";
  }
  const layers = [];
  for (let l = 0; l < dims.length - 1; l += 1) {
    layers.push(createLayer(dims[l], dims[l + 1], density));
  }
  return { layers, dims, outType };
}

function forward(model, x) {
  const acts = [x.slice()];
  const zs = [];
  for (let l = 0; l < model.layers.length; l += 1) {
    const layer = model.layers[l];
    const prev = acts[acts.length - 1];
    const z = new Array(layer.outDim).fill(0);
    for (let o = 0; o < layer.outDim; o += 1) {
      let acc = layer.b[o];
      for (let i = 0; i < layer.inDim; i += 1) {
        if (layer.m[o][i]) acc += layer.w[o][i] * prev[i];
      }
      z[o] = acc;
    }
    zs.push(z);
    if (l < model.layers.length - 1) {
      acts.push(z.map((v) => tanh(v)));
    } else if (model.outType === "binary") {
      acts.push([sigmoid(z[0])]);
    } else if (model.outType === "policy") {
      acts.push(softmax(z));
    } else {
      acts.push(z.slice());
    }
  }
  return { acts, zs };
}

function backwardAndUpdate(model, x, target, config) {
  const { lr, mode } = config;
  const cache = forward(model, x);
  const { acts, zs } = cache;
  const L = model.layers.length;
  const deltas = new Array(L);
  let loss = 0;
  let acc = 0;

  const out = acts[acts.length - 1];
  if (mode === "supervised") {
    const y = target[0];
    const p = clamp(out[0], 1e-6, 1 - 1e-6);
    loss = -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    acc = (p >= 0.5) === (y === 1) ? 1 : 0;
    deltas[L - 1] = [p - y];
  } else if (mode === "selfsupervised") {
    const e0 = out[0] - target[0];
    const e1 = out[1] - target[1];
    loss = 0.5 * (e0 * e0 + e1 * e1);
    const dist = Math.sqrt(e0 * e0 + e1 * e1);
    acc = dist < 0.18 ? 1 : 0;
    deltas[L - 1] = [e0, e1];
  } else {
    const action = target[0];
    const reward = target[1];
    const probs = out;
    const pAction = clamp(probs[action], 1e-6, 1);
    loss = -reward * Math.log(pAction);
    acc = reward;
    const d = probs.slice();
    d[action] -= 1;
    for (let i = 0; i < d.length; i += 1) d[i] *= reward;
    deltas[L - 1] = d;
  }

  for (let l = L - 1; l >= 0; l -= 1) {
    const layer = model.layers[l];
    const aPrev = acts[l];
    const delta = deltas[l];

    for (let o = 0; o < layer.outDim; o += 1) {
      layer.b[o] -= lr * delta[o];
      for (let i = 0; i < layer.inDim; i += 1) {
        const g = delta[o] * aPrev[i];
        layer.gradScore[o][i] = 0.95 * layer.gradScore[o][i] + 0.05 * Math.abs(g);
        if (layer.m[o][i]) layer.w[o][i] -= lr * g;
      }
    }

    if (l > 0) {
      const prevDelta = new Array(model.layers[l - 1].outDim).fill(0);
      for (let i = 0; i < model.layers[l - 1].outDim; i += 1) {
        let accDelta = 0;
        for (let o = 0; o < layer.outDim; o += 1) {
          if (layer.m[o][i]) accDelta += delta[o] * layer.w[o][i];
        }
        prevDelta[i] = accDelta * dtanh(zs[l - 1][i]);
      }
      deltas[l - 1] = prevDelta;
    }
  }

  return { loss, acc };
}

function createRuntime(mode) {
  if (mode === "supervised") {
    const xor = [
      { x: [0, 0], y: [0] },
      { x: [0, 1], y: [1] },
      { x: [1, 0], y: [1] },
      { x: [1, 1], y: [0] },
    ];
    return {
      batches: 24,
      lr: 0.06,
      sample() {
        const p = xor[Math.floor(seededRandom() * xor.length)];
        return { x: p.x, t: p.y };
      },
      evaluate(model) {
        let loss = 0;
        let acc = 0;
        for (const p of xor) {
          const out = forward(model, p.x).acts.at(-1)[0];
          const prob = clamp(out, 1e-6, 1 - 1e-6);
          loss += -(p.y[0] * Math.log(prob) + (1 - p.y[0]) * Math.log(1 - prob));
          acc += (prob >= 0.5) === (p.y[0] === 1) ? 1 : 0;
        }
        return { loss: loss / xor.length, acc: acc / xor.length };
      },
    };
  }

  if (mode === "selfsupervised") {
    return {
      batches: 30,
      lr: 0.03,
      sample() {
        const angle = seededRandom() * TAU;
        const radius = 0.35 + 0.55 * seededRandom();
        const clean = [radius * Math.cos(angle), radius * Math.sin(angle)];
        const noisy = [clean[0] + randn(0.09), clean[1] + randn(0.09)];
        return { x: noisy, t: clean };
      },
      evaluate(model) {
        let loss = 0;
        let acc = 0;
        const n = 40;
        for (let k = 0; k < n; k += 1) {
          const s = this.sample();
          const out = forward(model, s.x).acts.at(-1);
          const e0 = out[0] - s.t[0];
          const e1 = out[1] - s.t[1];
          const dist = Math.sqrt(e0 * e0 + e1 * e1);
          loss += 0.5 * (e0 * e0 + e1 * e1);
          acc += dist < 0.18 ? 1 : 0;
        }
        return { loss: loss / n, acc: acc / n };
      },
    };
  }

  return {
    batches: 36,
    lr: 0.03,
    sample() {
      const ctx = [seededRandom() * 2 - 1, seededRandom() * 2 - 1, 1];
      const means = [
        0.75 * ctx[0] - 0.2 * ctx[1],
        -0.4 * ctx[0] + 0.9 * ctx[1],
        0.25 * ctx[0] + 0.25 * ctx[1],
      ];
      const logits = forward(state.model, ctx).acts.at(-1);
      const r = seededRandom();
      let cum = 0;
      let action = 0;
      for (let i = 0; i < logits.length; i += 1) {
        cum += logits[i];
        if (r <= cum) {
          action = i;
          break;
        }
      }
      const noise = randn(0.15);
      const reward = means[action] + noise > 0 ? 1 : 0;
      return { x: ctx, t: [action, reward], expectedBest: Math.max(...means) };
    },
    evaluate(model) {
      let rewardMean = 0;
      let loss = 0;
      const n = 60;
      for (let k = 0; k < n; k += 1) {
        const s = this.sample();
        const probs = forward(model, s.x).acts.at(-1);
        const action = probs.indexOf(Math.max(...probs));
        const means = [
          0.75 * s.x[0] - 0.2 * s.x[1],
          -0.4 * s.x[0] + 0.9 * s.x[1],
          0.25 * s.x[0] + 0.25 * s.x[1],
        ];
        const rew = means[action] > 0 ? 1 : 0;
        rewardMean += rew;
        loss += -Math.log(clamp(probs[action], 1e-6, 1));
      }
      return { loss: loss / n, acc: rewardMean / n };
    },
  };
}

function rewireLayer(layer, method) {
  const active = [];
  const inactive = [];
  for (let o = 0; o < layer.outDim; o += 1) {
    for (let i = 0; i < layer.inDim; i += 1) {
      if (layer.m[o][i]) active.push([o, i]);
      else inactive.push([o, i]);
    }
  }
  if (active.length < 4 || inactive.length < 2) return [];

  const pruneCount = Math.max(1, Math.floor(active.length * 0.2));
  active.sort((a, b) => Math.abs(layer.w[a[0]][a[1]]) - Math.abs(layer.w[b[0]][b[1]]));
  const pruned = active.slice(0, pruneCount);
  for (const [o, i] of pruned) {
    layer.m[o][i] = false;
    layer.w[o][i] = 0;
  }

  let candidates = inactive.slice();
  if (method === "rigl") {
    candidates.sort((a, b) => layer.gradScore[b[0]][b[1]] - layer.gradScore[a[0]][a[1]]);
  } else {
    for (let k = candidates.length - 1; k > 0; k -= 1) {
      const r = Math.floor(seededRandom() * (k + 1));
      const t = candidates[k];
      candidates[k] = candidates[r];
      candidates[r] = t;
    }
  }
  const grown = [];
  for (let k = 0; k < pruneCount && k < candidates.length; k += 1) {
    const [o, i] = candidates[k];
    layer.m[o][i] = true;
    layer.w[o][i] = randn(0.1);
    layer.gradScore[o][i] = 0;
    grown.push([o, i]);
  }
  return grown;
}

function maybeRewire() {
  if (state.epoch === 0 || state.epoch % state.rewireEvery !== 0) return [];
  const changed = [];
  for (let l = 0; l < state.model.layers.length; l += 1) {
    const grown = rewireLayer(state.model.layers[l], state.method);
    for (const [o, i] of grown) changed.push([l, o, i]);
  }
  if (changed.length > 0) {
    state.phase = "rewire";
    state.lastRewireEpoch = state.epoch;
  }
  return changed;
}

function runEpoch() {
  if (state.epoch >= MAX_EPOCHS) return;
  const rt = state.runtime;
  let lossMean = 0;
  let accMean = 0;
  const phaseSeed = state.epoch % 4;
  state.phase = phaseSeed < 2 ? "feedforward" : "backprop";

  for (let b = 0; b < rt.batches; b += 1) {
    const s = rt.sample();
    const r = backwardAndUpdate(state.model, s.x, s.t, { lr: rt.lr, mode: state.mode });
    lossMean += r.loss;
    accMean += r.acc;
  }

  lossMean /= rt.batches;
  accMean /= rt.batches;
  state.epoch += 1;
  const rewireChanges = maybeRewire();
  if (rewireChanges.length > 0) state.rewireEpochs.push(state.epoch);
  const evalNow = rt.evaluate(state.model);
  state.history.push({
    epoch: state.epoch,
    loss: 0.45 * lossMean + 0.55 * evalNow.loss,
    acc: 0.35 * accMean + 0.65 * evalNow.acc,
  });
  if (state.history.length > 180) state.history.shift();
}

function countActiveConnections(model) {
  let active = 0;
  let total = 0;
  for (const layer of model.layers) {
    for (let o = 0; o < layer.outDim; o += 1) {
      for (let i = 0; i < layer.inDim; i += 1) {
        total += 1;
        if (layer.m[o][i]) active += 1;
      }
    }
  }
  return { active, total };
}

function computeNodeLayout() {
  const dims = state.model.dims;
  const w = ui.networkCanvas.width;
  const h = ui.networkCanvas.height;
  const marginX = 70;
  const marginY = 40;
  const xStep = (w - marginX * 2) / (dims.length - 1);
  const layers = [];
  for (let l = 0; l < dims.length; l += 1) {
    const n = dims[l];
    const yStep = (h - marginY * 2) / Math.max(1, n - 1);
    const nodes = [];
    for (let j = 0; j < n; j += 1) {
      nodes.push({
        x: marginX + xStep * l,
        y: marginY + yStep * j,
      });
    }
    layers.push(nodes);
  }
  return layers;
}

function drawNetwork() {
  const c = ui.networkCanvas;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  const layers = state.vis;
  const pulse = (Math.sin(state.pulseT * 0.01) + 1) * 0.5;
  const rewireGlow = state.epoch - state.lastRewireEpoch < 2 ? 1 : 0;

  for (let l = 0; l < state.model.layers.length; l += 1) {
    const layer = state.model.layers[l];
    const src = layers[l];
    const dst = layers[l + 1];
    for (let o = 0; o < layer.outDim; o += 1) {
      for (let i = 0; i < layer.inDim; i += 1) {
        if (!layer.m[o][i]) continue;
        const p0 = src[i];
        const p1 = dst[o];
        const baseAlpha = 0.2 + 0.45 * Math.min(1, Math.abs(layer.w[o][i]));
        let color = `rgba(130, 153, 220, ${baseAlpha})`;
        if (state.phase === "feedforward") {
          color = `rgba(90, 168, 255, ${baseAlpha + 0.35 * pulse})`;
        } else if (state.phase === "backprop") {
          color = `rgba(255, 107, 157, ${baseAlpha + 0.35 * pulse})`;
        } else if (state.phase === "rewire" && rewireGlow) {
          color = `rgba(255, 209, 102, ${0.45 + 0.45 * pulse})`;
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }
  }

  for (let l = 0; l < layers.length; l += 1) {
    for (const p of layers[l]) {
      ctx.fillStyle = "#d5def8";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.3, 0, TAU);
      ctx.fill();
      ctx.fillStyle = "#4d5f97";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, TAU);
      ctx.fill();
    }
  }
}

function drawMetrics() {
  const c = ui.metricsCanvas;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);

  const pad = { l: 56, r: 64, t: 16, b: 28 };
  const w = c.width - pad.l - pad.r;
  const h = c.height - pad.t - pad.b;
  const hist = state.history;
  const n = hist.length;
  if (n < 2) return;

  const minLoss = Math.min(...hist.map((d) => d.loss));
  const maxLoss = Math.max(...hist.map((d) => d.loss));
  const yLossMin = minLoss * 0.92;
  const yLossMax = maxLoss * 1.08 + 1e-6;

  function xScale(i) {
    return pad.l + (i / (n - 1)) * w;
  }
  function yLoss(v) {
    return pad.t + (1 - (v - yLossMin) / (yLossMax - yLossMin)) * h;
  }
  function yAcc(v) {
    return pad.t + (1 - v) * h;
  }

  ctx.strokeStyle = "rgba(120,140,200,0.45)";
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, c.height - pad.b);
  ctx.moveTo(c.width - pad.r, pad.t);
  ctx.lineTo(c.width - pad.r, c.height - pad.b);
  ctx.stroke();

  ctx.strokeStyle = "rgba(120,140,200,0.35)";
  ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k += 1) {
    const y = pad.t + (k / 4) * h;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(c.width - pad.r, y);
    ctx.stroke();
  }

  const rewireSet = new Set(state.rewireEpochs);
  ctx.strokeStyle = "rgba(255, 209, 102, 0.45)";
  ctx.lineWidth = 1.2;
  for (let i = 0; i < n; i += 1) {
    const p = hist[i];
    if (!rewireSet.has(p.epoch)) continue;
    const x = xScale(i);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, c.height - pad.b);
    ctx.stroke();
  }

  ctx.strokeStyle = "#4be38a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const p = hist[i];
    const x = xScale(i);
    const y = yLoss(p.loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  for (let i = 0; i < n; i += 1) {
    const p = hist[i];
    if (!rewireSet.has(p.epoch)) continue;
    const x = xScale(i);
    const yL = yLoss(p.loss);
    const yA = yAcc(clamp(p.acc, 0, 1));

    ctx.fillStyle = "#ffd166";
    ctx.beginPath();
    ctx.arc(x, yL, 3, 0, TAU);
    ctx.fill();

    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, yA, 4, 0, TAU);
    ctx.stroke();
  }

  ctx.strokeStyle = "#5aa8ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const p = hist[i];
    const x = xScale(i);
    const y = yAcc(clamp(p.acc, 0, 1));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const lossLast = hist[n - 1].loss;
  const accLast = clamp(hist[n - 1].acc, 0, 1);
  const lossTipX = xScale(n - 1);
  const lossTipY = yLoss(lossLast);
  const accTipX = xScale(n - 1);
  const accTipY = yAcc(accLast);

  function drawTip(x, y, text, color) {
    const tx = Math.min(c.width - 48, x + 8);
    const ty = clamp(y - 8, 16, c.height - 14);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, TAU);
    ctx.fill();
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(text).width;
    const bw = tw + 10;
    const bh = 18;
    ctx.fillStyle = "rgba(9,12,20,0.9)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.fillRect(tx, ty - bh, bw, bh);
    ctx.strokeRect(tx, ty - bh, bw, bh);
    ctx.fillStyle = color;
    ctx.fillText(text, tx + 5, ty - 5);
  }

  drawTip(lossTipX, lossTipY, lossLast.toFixed(3), "#4be38a");
  drawTip(accTipX, accTipY, `${(accLast * 100).toFixed(1)}%`, "#5aa8ff");

  ctx.font = "11px sans-serif";
  for (let k = 0; k <= 4; k += 1) {
    const t = k / 4;
    const y = pad.t + t * h;
    const lossVal = yLossMax - t * (yLossMax - yLossMin);
    const accVal = (1 - t) * 100;
    ctx.fillStyle = "#98e4b7";
    ctx.fillText(lossVal.toFixed(2), 6, y + 4);
    ctx.fillStyle = "#8ec0ff";
    const rightLabel = `${accVal.toFixed(0)}%`;
    const rw = ctx.measureText(rightLabel).width;
    ctx.fillText(rightLabel, c.width - rw - 6, y + 4);
  }

  ctx.fillStyle = "#b9c8ef";
  ctx.font = "12px sans-serif";
  ctx.fillText("Loss (left axis)", 8, 20);
  ctx.fillStyle = "#8ec0ff";
  ctx.fillText("Accuracy / Reward (right axis)", 8, 36);
  ctx.fillStyle = "#ffd166";
  ctx.fillText("Rewire events", 8, 52);
}

function updateReadout() {
  ui.epochReadout.textContent = `Epoch ${state.epoch}`;
  if (state.autoStoppedAtPerfect) {
    ui.phaseBadge.textContent = "Auto-stopped at 100%";
  } else {
    ui.phaseBadge.textContent =
      state.phase === "rewire" ? "Rewire Event" : state.phase[0].toUpperCase() + state.phase.slice(1);
  }
  const c = countActiveConnections(state.model);
  const dens = ((c.active / c.total) * 100).toFixed(1);
  const last = state.history.at(-1) || { loss: 0, acc: 0 };
  const metricName = state.mode === "reinforcement" ? "Reward" : "Accuracy";
  ui.statsReadout.textContent =
    `Method: ${state.method.toUpperCase()}  |  Active: ${c.active}/${c.total} (${dens}% density)  |  Loss: ${last.loss.toFixed(
      4
    )}  |  ${metricName}: ${(last.acc * 100).toFixed(1)}%`;
}

function frame() {
  state.pulseT += 1 + state.speed * 0.6;
  drawNetwork();
  drawMetrics();
  updateReadout();
  requestAnimationFrame(frame);
}

function stepOnce() {
  if (state.epoch >= MAX_EPOCHS) {
    state.phase = "completed";
    pause();
    return;
  }
  if (reachedPerfectAccuracy()) {
    state.autoStoppedAtPerfect = true;
    state.phase = "completed";
    pause();
    return;
  }
  runEpoch();
  if (reachedPerfectAccuracy()) {
    state.autoStoppedAtPerfect = true;
    state.phase = "completed";
    pause();
  }
}

function play() {
  if (state.running) return;
  if (state.epoch >= MAX_EPOCHS) {
    state.phase = "completed";
    updateReadout();
    return;
  }
  if (reachedPerfectAccuracy()) {
    state.autoStoppedAtPerfect = true;
    state.phase = "completed";
    updateReadout();
    return;
  }
  state.running = true;
  const base = 900;
  const period = Math.max(140, base - state.speed * 150);
  state.timer = setInterval(stepOnce, period);
}

function pause() {
  state.running = false;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function rebuildModel() {
  pause();
  resetSeed();
  state.autoStoppedAtPerfect = false;
  state.epoch = 0;
  state.history = [];
  state.rewireEpochs = [];
  state.lastRewireEpoch = -999;
  state.phase = "idle";
  state.model = createModel(state.mode, state.sparsity);
  state.runtime = createRuntime(state.mode);
  state.vis = computeNodeLayout();
}

function syncFromInputs() {
  state.mode = ui.mode.value;
  state.method = ui.method.value;
  state.sparsity = Number(ui.sparsity.value) / 100;
  state.rewireEvery = clamp(Number(ui.rewireEvery.value) || 8, 1, 40);
  state.speed = Number(ui.speed.value) || 3;
  ui.sparsityValue.textContent = `${Math.round(state.sparsity * 100)}%`;
}

ui.play.addEventListener("click", () => {
  syncFromInputs();
  play();
});
ui.pause.addEventListener("click", pause);
ui.step.addEventListener("click", () => {
  syncFromInputs();
  stepOnce();
});
ui.reset.addEventListener("click", () => {
  syncFromInputs();
  rebuildModel();
});

ui.mode.addEventListener("change", () => {
  syncFromInputs();
  rebuildModel();
});
ui.method.addEventListener("change", syncFromInputs);
ui.sparsity.addEventListener("input", () => {
  syncFromInputs();
  rebuildModel();
});
ui.rewireEvery.addEventListener("change", syncFromInputs);
ui.speed.addEventListener("input", () => {
  syncFromInputs();
  if (state.running) {
    pause();
    play();
  }
});

syncFromInputs();
rebuildModel();
for (let i = 0; i < 4; i += 1) stepOnce();
frame();

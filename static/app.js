const state = {
  metadata: null,
  values: {},
  activePreset: "",
};

const els = {
  health: document.querySelector("#healthStatus"),
  form: document.querySelector("#predictionForm"),
  controls: document.querySelector("#featureControls"),
  presets: document.querySelector("#presetStrip"),
  tier: document.querySelector("#tierName"),
  probability: document.querySelector("#probabilityValue"),
  quality: document.querySelector("#qualityValue"),
  accuracy: document.querySelector("#accuracyValue"),
  roc: document.querySelector("#rocValue"),
  meter: document.querySelector("#qualityMeterFill"),
  notes: document.querySelector("#tastingNotes"),
  importance: document.querySelector("#importanceList"),
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatMetric(value) {
  return Number(value).toFixed(2);
}

function inputId(name) {
  return `feature-${name.replace(/[^a-z0-9]+/gi, "-")}`;
}

function setHealth(text) {
  els.health.textContent = text;
}

function setLoading(isLoading) {
  document.body.classList.toggle("is-loading", isLoading);
}

function renderPresets() {
  els.presets.innerHTML = "";
  state.metadata.presets.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-button";
    button.textContent = preset.name;
    button.dataset.preset = preset.name;
    button.addEventListener("click", () => applyPreset(preset));
    els.presets.appendChild(button);
  });
}

function renderControls() {
  els.controls.innerHTML = "";
  state.metadata.featureSpecs.forEach((feature) => {
    state.values[feature.name] = feature.median;

    const row = document.createElement("div");
    row.className = "feature-row";

    const id = inputId(feature.name);
    const label = document.createElement("label");
    label.setAttribute("for", id);
    label.textContent = feature.label;

    const range = document.createElement("input");
    range.type = "range";
    range.id = id;
    range.name = feature.name;
    range.min = feature.min;
    range.max = feature.max;
    range.step = feature.step;
    range.value = feature.median;

    const number = document.createElement("input");
    number.type = "number";
    number.min = feature.min;
    number.max = feature.max;
    number.step = feature.step;
    number.value = feature.median;
    number.dataset.feature = feature.name;

    range.addEventListener("input", () => {
      number.value = range.value;
      state.values[feature.name] = Number(range.value);
    });

    number.addEventListener("input", () => {
      range.value = number.value;
      state.values[feature.name] = Number(number.value);
    });

    row.append(label, range, number);
    els.controls.appendChild(row);
  });
}

function renderMetrics() {
  const metrics = state.metadata.metrics;
  els.accuracy.textContent = formatPercent(metrics.accuracy);
  els.roc.textContent = formatMetric(metrics.rocAuc);
}

function renderImportance() {
  els.importance.innerHTML = "";
  const top = state.metadata.featureImportance.slice(0, 7);
  const max = Math.max(...top.map((item) => item.importance));

  top.forEach((item) => {
    const row = document.createElement("div");
    row.className = "importance-row";

    const name = document.createElement("span");
    name.textContent = item.label;

    const track = document.createElement("div");
    track.className = "importance-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(6, (item.importance / max) * 100)}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.textContent = formatMetric(item.importance);

    row.append(name, track, value);
    els.importance.appendChild(row);
  });
}

function applyPreset(preset) {
  state.activePreset = preset.name;
  state.values = { ...preset.values };

  state.metadata.featureSpecs.forEach((feature) => {
    const range = document.querySelector(`#${inputId(feature.name)}`);
    const number = document.querySelector(`[data-feature="${feature.name}"]`);
    const value = preset.values[feature.name];
    if (range && number && value !== undefined) {
      range.value = value;
      number.value = value;
    }
  });

  document.querySelectorAll(".preset-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.preset === preset.name);
  });

  evaluate();
}

function renderPrediction(result) {
  const pct = Math.round(result.highQualityProbability * 100);
  els.tier.textContent = result.tier;
  els.probability.textContent = `${pct}%`;
  els.quality.textContent = result.predictedQuality.toFixed(2);
  els.meter.style.width = `${pct}%`;

  els.notes.innerHTML = "";
  result.notes.forEach((note) => {
    const node = document.createElement("div");
    node.className = "note";
    node.textContent = note;
    els.notes.appendChild(node);
  });
}

async function evaluate() {
  try {
    setLoading(true);
    const result = await fetchJson("/api/predict", {
      method: "POST",
      body: JSON.stringify(state.values),
    });
    renderPrediction(result);
    setHealth("Model ready");
  } catch (error) {
    setHealth(error.message);
  } finally {
    setLoading(false);
  }
}

async function init() {
  try {
    setHealth("Training cellar");
    state.metadata = await fetchJson("/api/metadata");
    renderPresets();
    renderControls();
    renderMetrics();
    renderImportance();
    const preferred = state.metadata.presets.find((preset) => preset.name === "Private Reserve");
    applyPreset(preferred || state.metadata.presets[0]);
  } catch (error) {
    setHealth(error.message);
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  evaluate();
});

init();


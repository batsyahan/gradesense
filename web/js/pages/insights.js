import { api } from "../api.js";
import { lineChart } from "../charts.js";
import { comparisonPanel, confusionPanel, splitTablePanel } from "../components/analytics.js";
import { state } from "../state.js";
import { $, $$, alertBox, esc, fmt, icon, loading, num, refreshIcons } from "../ui.js";

const METRICS = { macro_f1: "Macro-F1", weighted_f1: "Weighted-F1", accuracy: "Accuracy" };

export async function render(view) {
  const m = state.meta.model;
  view.innerHTML = `
    <div class="page-head"><div><h1>Model Insights</h1>
      <p>How every model performs, read directly from the saved evaluation results.</p></div></div>
    <div class="grid grid-2">
      <section class="card card-pad"><h2>Model</h2><dl class="kv mt-sm">
        <dt>Architecture</dt><dd>DeBERTa-v3-base cross-encoder${m.parameters ? ` (${num(m.parameters)} parameters)` : ""}</dd>
        <dt>Input</dt><dd><code>[CLS] question + reference [SEP] student answer [SEP]</code>, max ${num(m.max_length)} tokens</dd>
        <dt>Fine-tuning</dt><dd>AdamW, lr 2e-5, batch 16, 10 epochs, linear warmup, class-weighted loss; best epoch chosen on validation macro-F1</dd>
        <dt>Status</dt><dd>${m.status === "loaded" ? `<span class="badge green">Loaded</span> on ${esc(m.device)}${m.load_seconds ? ` in ${m.load_seconds}s` : ""}`
          : m.status === "error" ? `<span class="badge red">Error</span> ${esc(m.error)}` : `<span class="badge amber">${esc(m.status)}</span>`}</dd>
      </dl></section>
      <section class="card card-pad"><h2>Dataset</h2><dl class="kv mt-sm">
        <dt>Name</dt><dd>SciEntsBank (SemEval-2013 Task 7)</dd>
        <dt>Labels</dt><dd>5-way: correct, partially correct, contradictory, irrelevant, non-answer</dd>
        <dt>Training data</dt><dd>4,472 training and 497 validation answers (10% stratified, seed 42)</dd>
        <dt>Test splits</dt><dd>Unseen Answers (540) · Unseen Questions (733) · Unseen Domains (4,562)</dd>
      </dl></section>
    </div>
    <section class="card card-pad mt" id="i-compare"></section>
    <section class="card card-pad mt" id="i-table"></section>
    <div class="grid grid-2 mt">
      <section class="card card-pad" id="i-confusion"></section>
      <section class="card card-pad" id="i-splits"></section>
    </div>
    <section class="card card-pad mt" id="i-training"></section>
    <section class="card card-pad mt" id="i-status"></section>`;

  const cmp = await comparisonPanel($("#i-compare", view), { chartOnly: false, controls: true, title: "Baseline Comparison" });
  const models = cmp ? cmp.models.filter((x) => x.evaluated) : [];
  fullTable($("#i-table", view), "macro_f1");
  confusionPanel($("#i-confusion", view), { models: models.length ? models : null });
  modelSplitTable($("#i-splits", view), models);
  training($("#i-training", view));
  status($("#i-status", view));
}

async function fullTable(el, metric) {
  el.innerHTML = `<div class="card-head"><div><h2>Split-wise evaluation</h2><p>Every model on validation and all three test splits. Best per split in green.</p></div>
    <div class="segmented" role="group" aria-label="Metric">${Object.entries(METRICS).map(([k, n]) =>
      `<button type="button" data-m="${k}" aria-pressed="${k === metric}">${n}</button>`).join("")}</div></div><div id="t-body">${loading()}</div>`;
  $$("[data-m]", el).forEach((b) => b.addEventListener("click", () => fullTable(el, b.dataset.m)));
  let data;
  try { data = await api("/api/results/comparison", { params: { metric } }); }
  catch (e) { $("#t-body", el).innerHTML = alertBox("error", esc(e.message)); refreshIcons(); return; }
  const splits = data.splits;
  const best = {};
  splits.forEach((s) => { best[s.key] = Math.max(...data.models.map((mm) => mm.values[s.key] ?? -1)); });
  $("#t-body", el).innerHTML = `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Family</th>${splits.map((s) => `<th class="num">${esc(s.name)}</th>`).join("")}</tr></thead>
    <tbody>${data.models.map((mm) => `<tr><td><strong>${esc(mm.name)}</strong></td><td>${esc(mm.family)}</td>${splits.map((s) => {
      const v = mm.values[s.key];
      if (v === null) return `<td class="num na">Not evaluated</td>`;
      return `<td class="num ${v === best[s.key] ? "best" : ""}">${fmt(v, 3)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="help mt-sm">Validation scores are optimistic: the best epoch/threshold was chosen on validation. Compare models on the test splits.</p>`;
}

function modelSplitTable(el, models) {
  const pick = models.find((x) => x.key === "DeBERTa-v3") || models[0];
  if (!pick) { splitTablePanel(el); return; }
  const draw = (key) => {
    splitTablePanel(el, { model: key, includeVal: true }).then(() => {
      const head = $(".card-head", el);
      const sel = document.createElement("select");
      sel.setAttribute("aria-label", "Model");
      sel.style.width = "auto";
      sel.innerHTML = models.map((x) => `<option value="${esc(x.key)}" ${x.key === key ? "selected" : ""}>${esc(x.name)}</option>`).join("");
      sel.addEventListener("change", () => draw(sel.value));
      head.appendChild(sel);
    });
  };
  draw(pick.key);
}

async function training(el) {
  el.innerHTML = `<div class="card-head"><div><h2>DeBERTa-v3 training curve</h2><p id="tr-sub"></p></div></div><div id="tr-body">${loading()}</div>`;
  let h;
  try { h = await api("/api/results/training"); }
  catch (e) { $("#tr-body", el).innerHTML = alertBox("error", esc(e.message)); refreshIcons(); return; }
  if (!h.available) {
    $("#tr-body", el).innerHTML = `<p class="help">No training history found (<code>results/history_deberta_5way.csv</code>). It is written by the Colab training notebook.</p>`;
    return;
  }
  $("#tr-sub", el).textContent = `Best epoch: ${h.best_epoch} (highest validation macro-F1). Later epochs keep lowering training loss while validation stops improving: overfitting, which is why the best epoch was kept.`;
  $("#tr-body", el).innerHTML = `<div class="chart-box"><canvas role="img" aria-label="Training loss and validation macro-F1 per epoch"></canvas></div>`;
  lineChart($("canvas", el), {
    labels: h.epochs.map((e) => `Epoch ${e}`),
    series: [{ label: "Training loss", data: h.train_loss, color: "#2563eb", axis: "y" },
             { label: "Validation macro-F1", data: h.val_macro_f1, color: "#16a34a", axis: "y2" }],
    yTitle: "Training loss",
  });
}

async function status(el) {
  el.innerHTML = `<h2>Evaluation run status</h2><div id="s-body" class="mt-sm">${loading()}</div>`;
  let d;
  try { d = await api("/api/results/status"); }
  catch (e) { $("#s-body", el).innerHTML = alertBox("error", esc(e.message)); refreshIcons(); return; }
  const size = (b) => b === null || b === undefined ? "" : b > 1e6 ? `${(b / 1e6).toFixed(0)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`;
  $("#s-body", el).innerHTML = `<div class="table-wrap"><table><thead><tr><th>Artifact</th><th>Used for</th><th>Status</th><th>File</th></tr></thead><tbody>
    ${d.artifacts.map((a) => `<tr><td>${esc(a.name)}</td><td>${esc(a.purpose)}</td>
      <td>${a.exists ? `<span class="badge green">${icon("circle-check")}Present</span>` : `<span class="badge gray">Missing</span>`}</td>
      <td><code>${esc(a.path)}</code>${a.exists && (a.size || a.modified) ? `<br><small class="help">${size(a.size)} ${a.modified ? "· " + esc(a.modified) : ""}</small>` : ""}</td></tr>`).join("")}
    </tbody></table></div>
    <p class="help mt-sm">Missing baseline files are created by <code>python scripts/build_baselines.py</code>. Missing DeBERTa files come from the Colab notebook.</p>`;
  refreshIcons();
}

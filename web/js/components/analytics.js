// The three analytics panels: model comparison chart, confusion matrix, split results table.
import { api } from "../api.js";
import { groupedBar } from "../charts.js";
import { $, $$, alertBox, emptyState, esc, fmt, loading, num, refreshIcons } from "../ui.js";

const SPLITS = [["test_ua", "Unseen Answers"], ["test_uq", "Unseen Questions"], ["test_ud", "Unseen Domains"]];
const METRIC_NAMES = { macro_f1: "Macro-F1", weighted_f1: "Weighted-F1", accuracy: "Accuracy" };
const SPLIT_COLORS = ["#1e3a8a", "#3b82f6", "#93c5fd"];

// ------------------------------------------------------------------ comparison chart
export async function comparisonPanel(el, { metric = "macro_f1", chartOnly = true, title = "Model Performance Comparison", controls = false } = {}) {
  el.innerHTML = `<div class="card-head"><div><h2>${esc(title)}</h2><p id="cmp-sub"></p></div>
    ${controls ? `<div class="segmented" role="group" aria-label="Metric">${Object.entries(METRIC_NAMES).map(([k, n]) =>
      `<button type="button" data-metric="${k}" aria-pressed="${k === metric}">${n}</button>`).join("")}</div>` : ""}</div>
    <div id="cmp-body">${loading("Loading results…")}</div>`;
  if (controls) {
    $$("[data-metric]", el).forEach((b) => b.addEventListener("click", () =>
      comparisonPanel(el, { metric: b.dataset.metric, chartOnly, title, controls })));
  }
  const body = $("#cmp-body", el);
  let data;
  try {
    data = await api("/api/results/comparison", { params: { metric, chart_only: chartOnly } });
  } catch (e) {
    body.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return null;
  }
  const evaluated = data.models.filter((m) => m.evaluated);
  if (!evaluated.length) {
    body.innerHTML = emptyState("No evaluation results yet",
      "Run the training scripts so <code>results/results.csv</code> exists, then reload.", "chart-column");
    refreshIcons();
    return data;
  }
  const notEval = data.models.filter((m) => !m.evaluated).map((m) => m.short);
  $("#cmp-sub", el).textContent = `${METRIC_NAMES[metric]} on the three SciEntsBank test splits (5-way labels).`;
  body.innerHTML = `<div class="chart-box"><canvas aria-label="Grouped bar chart of ${METRIC_NAMES[metric]} by model and test split" role="img"></canvas></div>
    ${notEval.length ? `<p class="help mt-sm">Not evaluated: ${esc(notEval.join(", "))} - no results exist for ${notEval.length === 1 ? "this model" : "these models"}, so no bars are shown.</p>` : ""}`;
  groupedBar($("canvas", body), {
    labels: data.models.map((m) => m.evaluated ? m.short : `${m.short} (not evaluated)`),
    series: SPLITS.map(([k, n], i) => ({ label: n, data: data.models.map((m) => m.values[k]), color: SPLIT_COLORS[i] })),
    yTitle: METRIC_NAMES[metric],
  });
  return data;
}

// ------------------------------------------------------------------ confusion matrix
export async function confusionPanel(el, { model = "DeBERTa-v3", split = "test_ua", mode = "counts", models = null, title = "Confusion Matrix (5-way)" } = {}) {
  const opts = { model, split, mode, models, title };
  el.innerHTML = `<div class="card-head"><div><h2>${esc(title)}</h2><p id="cm-sub"></p></div></div>
    <div class="controls" style="margin-bottom:12px">
      ${models ? `<label class="sr-only" for="cm-model">Model</label><select id="cm-model">${models.map((m) =>
        `<option value="${esc(m.key)}" ${m.key === model ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select>` : ""}
      <label class="sr-only" for="cm-split">Test split</label>
      <select id="cm-split">${SPLITS.map(([k, n]) => `<option value="${k}" ${k === split ? "selected" : ""}>${n}</option>`).join("")}</select>
      <div class="segmented" role="group" aria-label="Show counts or percentages">
        <button type="button" data-mode="counts" aria-pressed="${mode === "counts"}">Counts</button>
        <button type="button" data-mode="normalized" aria-pressed="${mode === "normalized"}">Row %</button>
      </div>
    </div>
    <div id="cm-body">${loading()}</div>`;
  const rerender = (change) => confusionPanel(el, { ...opts, ...change });
  if (models) $("#cm-model", el).addEventListener("change", (e) => rerender({ model: e.target.value }));
  $("#cm-split", el).addEventListener("change", (e) => rerender({ split: e.target.value }));
  $$("[data-mode]", el).forEach((b) => b.addEventListener("click", () => rerender({ mode: b.dataset.mode })));

  const body = $("#cm-body", el);
  let cm;
  try {
    cm = await api("/api/results/confusion", { params: { model, split } });
  } catch (e) {
    body.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  if (!cm.available) {
    body.innerHTML = emptyState("No predictions saved", esc(cm.message), "grid-3x3");
    refreshIcons();
    return;
  }
  $("#cm-sub", el).textContent = `${cm.model.name} · ${cm.split_name} · ${num(cm.total)} answers`;
  const short = ["Correct", "Contradictory", "Partial", "Irrelevant", "Non-Answer"];
  const abbrev = ["Cor.", "Cont.", "Part.", "Irr.", "Non."];
  const rows = cm.counts.map((row, i) => `<tr><th class="row-h" scope="row">${short[i]}</th>${row.map((c, j) => {
    const frac = cm.normalized[i][j];
    const bg = `rgba(37, 99, 235, ${(0.06 + frac * 0.94).toFixed(3)})`;
    const color = frac > 0.5 ? "#fff" : "#1e293b";
    const text = mode === "counts" ? num(c) : `${(frac * 100).toFixed(0)}%`;
    return `<td class="cell" style="background:${bg};color:${color}" title="True ${short[i]}, predicted ${short[j]}: ${num(c)} answers (${(frac * 100).toFixed(1)}% of row)">${text}</td>`;
  }).join("")}<td class="cell" style="color:var(--muted)">${num(cm.row_totals[i])}</td></tr>`).join("");
  body.innerHTML = `<div style="overflow-x:auto"><table class="heatmap" aria-label="Confusion matrix: rows are true labels, columns are predicted labels">
      <tbody>${rows}<tr><th></th>${abbrev.map((a, i) => `<th class="col-h" scope="col" title="${short[i]}">${a}</th>`).join("")}<th class="col-h">Total</th></tr></tbody>
    </table></div>
    <div class="axis-label">Rows: true label · Columns: predicted label</div>
    <div class="scale"><span>0%</span><span class="scale-bar"></span><span>100% of row</span></div>
    <p class="help mt-sm">${mode === "counts" ? "Cells show answer counts; colour shows the share of each true-label row." : "Cells show the share of each true-label row (recall per class on the diagonal)."}</p>`;
}

// ------------------------------------------------------------------ split results table
export async function splitTablePanel(el, { model = "DeBERTa-v3", title = null, includeVal = false } = {}) {
  el.innerHTML = `<div class="card-head"><div><h2 id="st-title">${esc(title || "Test Split Results")}</h2><p id="st-sub"></p></div></div><div id="st-body">${loading()}</div>`;
  const body = $("#st-body", el);
  let data;
  try {
    data = await api("/api/results/splits", { params: { model } });
  } catch (e) {
    body.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  if (!title) $("#st-title", el).textContent = `Test Split Results (${data.model.short})`;
  $("#st-sub", el).textContent = data.averaging;
  const rows = data.rows.filter((r) => includeVal || r.split !== "val");
  if (rows.every((r) => r.macro_f1 === null)) {
    body.innerHTML = emptyState("Not evaluated", `No results exist for ${esc(data.model.name)} yet.`, "table");
    refreshIcons();
    return;
  }
  const cell = (v) => v === null ? `<td class="num na">Not evaluated</td>` : `<td class="num">${fmt(v, 3)}</td>`;
  body.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Split</th><th class="num">Macro-F1</th><th class="num">Accuracy</th><th class="num">Precision</th><th class="num">Recall</th></tr></thead>
    <tbody>${rows.map((r) => `<tr><td>${esc(r.split_name)}${r.support ? `<br><small class="help">${num(r.support)} answers</small>` : ""}</td>${cell(r.macro_f1)}${cell(r.accuracy)}${cell(r.precision)}${cell(r.recall)}</tr>`).join("")}</tbody>
    </table></div>
    ${rows.some((r) => r.precision === null && r.macro_f1 !== null) ? `<p class="help mt-sm">Precision and recall need saved predictions; they show "Not evaluated" where only summary scores exist.</p>` : ""}`;
}

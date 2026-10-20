import { api } from "../api.js";
import { comparisonPanel, confusionPanel, splitTablePanel } from "../components/analytics.js";
import { renderWorkspace } from "../components/workspace.js";
import { $, esc, fmt, icon, num, refreshIcons, sparkline } from "../ui.js";

export async function render(view) {
  view.innerHTML = `
    <div class="page-head"><div><h1>Welcome to GradeSense</h1>
      <p>Automatic answer grading with transformer models and detailed feedback</p></div></div>
    <div class="grid grid-4" id="cards">${[1, 2, 3, 4].map(() => `<div class="card stat"><div class="stat-body"><div class="stat-label">&nbsp;</div><div class="stat-value">&nbsp;</div></div></div>`).join("")}</div>
    <div class="mt" id="workspace"></div>
    <div class="grid grid-analytics mt">
      <section class="card card-pad" id="p-compare"></section>
      <section class="card card-pad" id="p-confusion"></section>
      <section class="card card-pad" id="p-splits"></section>
    </div>`;

  renderWorkspace($("#workspace", view));
  loadCards($("#cards", view));
  comparisonPanel($("#p-compare", view));
  confusionPanel($("#p-confusion", view));
  splitTablePanel($("#p-splits", view));
}

async function loadCards(el) {
  let d;
  try {
    d = await api("/api/dashboard");
  } catch (e) {
    el.innerHTML = `<div class="card card-pad" style="grid-column:1/-1">${esc(e.message)}</div>`;
    return;
  }
  const h = d.history;
  const best = d.best_model;
  const deb = d.deberta.rows.filter((r) => r.split !== "val");
  const evaluatedSplits = deb.filter((r) => r.macro_f1 !== null);
  const ua = deb.find((r) => r.split === "test_ua");

  // Card 1 - total answers graded (from the history database)
  let c1;
  if (!h.enabled) {
    c1 = `<div class="stat-value sm">Not recorded</div><div class="stat-note">Grading history is turned off on this server.</div>`;
  } else {
    const series = h.daily.map((x) => x.count);
    const trend = h.week_change === null ? "" :
      `<span class="${h.week_change >= 0 ? "trend-up" : "trend-down"}">${h.week_change >= 0 ? "↑" : "↓"} ${Math.abs(h.week_change * 100).toFixed(0)}% vs last week</span>`;
    c1 = `<div class="stat-row"><div><div class="stat-value">${num(h.total)}</div>
      <div class="stat-note">${trend || `${num(h.this_week)} in the last 7 days`}</div></div>${sparkline(series, "#16a34a")}</div>`;
  }

  // Card 2 - DeBERTa macro-F1 on unseen answers (never labelled as accuracy)
  const c2 = ua && ua.macro_f1 !== null
    ? `<div class="stat-value">${fmt(ua.macro_f1, 3)}</div><div class="stat-note">Macro-F1 · DeBERTa-v3 · Unseen Answers</div>`
    : `<div class="stat-value sm">Not evaluated</div><div class="stat-note">No DeBERTa results in results/ yet.</div>`;

  // Card 3 - test splits evaluated
  const c3 = `<div class="stat-value">${evaluatedSplits.length}</div><div class="stat-note">${
    evaluatedSplits.length ? esc(evaluatedSplits.map((r) => r.split_name).join(" · ")) : "No test split evaluated yet"}</div>`;

  // Card 4 - best model + loaded / evaluated status
  const loaded = d.model.status === "loaded";
  const status = loaded ? `<span class="badge green">Loaded</span>`
    : d.model.status === "loading" ? `<span class="badge amber">Loading</span>` : `<span class="badge red">Not loaded</span>`;
  const c4 = best
    ? `<div class="stat-value sm">${esc(best.short)}</div><div class="stat-note">Evaluated · mean test macro-F1 ${fmt(best.mean_macro_f1, 3)}</div>`
    : `<div class="stat-value sm">Not evaluated</div><div class="stat-note">No results to compare yet.</div>`;

  el.innerHTML = `
    <div class="card stat"><div class="stat-icon">${icon("file-text")}</div><div class="stat-body"><div class="stat-label">Total Answers Graded</div>${c1}</div></div>
    <div class="card stat"><div class="stat-icon green">${icon("circle-check")}</div><div class="stat-body"><div class="stat-label">Grading Model Macro-F1</div>${c2}</div></div>
    <div class="card stat"><div class="stat-icon">${icon("layers")}</div><div class="stat-body"><div class="stat-label">Test Splits Evaluated</div>${c3}</div></div>
    <div class="card stat"><div class="stat-icon">${icon("chart-column")}</div><div class="stat-body"><div class="stat-label"><span>Best Model</span>${status}</div>${c4}</div></div>`;
  refreshIcons();
}

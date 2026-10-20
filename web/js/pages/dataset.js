import { api } from "../api.js";
import { simpleBar } from "../charts.js";
import { $, $$, alertBox, debounce, emptyState, esc, labelBadge, loading, num, pager, refreshIcons } from "../ui.js";

const LABEL_COLORS = ["#16a34a", "#dc2626", "#d97706", "#64748b", "#94a3b8"];

export async function render(view, params) {
  view.innerHTML = `
    <div class="page-head"><div><h1>Dataset Explorer</h1>
      <p>SciEntsBank, split exactly as the models were trained and evaluated.</p></div></div>
    <section class="card card-pad" id="d-overview">${loading("Loading the dataset (first time downloads ~0.5 MB)…")}</section>
    <div class="grid grid-2 mt">
      <section class="card card-pad" id="d-dist"></section>
      <section class="card card-pad" id="d-quality"></section>
    </div>
    <section class="card card-pad mt" id="d-samples"></section>`;

  let data;
  try {
    data = await api("/api/dataset/overview");
  } catch (e) {
    $("#d-overview", view).innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  const splits = data.splits;
  overview($("#d-overview", view), splits);
  distribution($("#d-dist", view), splits, "train");
  quality($("#d-quality", view), splits);
  samples($("#d-samples", view), splits, { split: params.split || "train", label: "", q: params.q || "", offset: 0 });
}

function overview(el, splits) {
  const roleBadge = (r) => r === "train" ? `<span class="badge blue">Training</span>` : r === "validation"
    ? `<span class="badge amber">Validation</span>` : `<span class="badge gray">Test (held out)</span>`;
  el.innerHTML = `<div class="card-head"><div><h2>Splits</h2><p>Training data fits the model, validation picks the best epoch, the three test splits are never used for training or model selection.</p></div></div>
    <div class="table-wrap"><table><thead><tr><th>Split</th><th>Role</th><th class="num">Answers</th><th class="num">Questions</th><th>What it measures</th></tr></thead>
    <tbody>${splits.map((s) => `<tr><td><strong>${esc(s.name)}</strong></td><td>${roleBadge(s.role)}</td><td class="num">${num(s.rows)}</td>
      <td class="num">${num(s.questions)}</td><td>${esc(s.description)}</td></tr>`).join("")}</tbody></table></div>`;
}

function distribution(el, splits, key) {
  const s = splits.find((x) => x.key === key);
  el.innerHTML = `<div class="card-head"><div><h2>Label distribution</h2><p>${num(s.rows)} answers in ${esc(s.name)}</p></div>
    <select aria-label="Split" style="width:auto">${splits.map((x) => `<option value="${x.key}" ${x.key === key ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select></div>
    <div class="chart-box short"><canvas role="img" aria-label="Number of answers per label"></canvas></div>`;
  $("select", el).addEventListener("change", (e) => distribution(el, splits, e.target.value));
  simpleBar($("canvas", el), {
    labels: s.distribution.map((d) => d.name), data: s.distribution.map((d) => d.count), colors: LABEL_COLORS, yTitle: "Answers",
  });
}

function quality(el, splits) {
  el.innerHTML = `<h2>Class imbalance and missing data</h2>
    <div class="table-wrap mt-sm"><table><thead><tr><th>Split</th><th class="num">Largest / smallest class</th><th>Absent labels</th><th class="num">Empty fields</th><th class="num">Answer words (mean / 95th pct)</th></tr></thead>
    <tbody>${splits.map((s) => {
      const empty = Object.values(s.missing).reduce((a, b) => a + b, 0);
      return `<tr><td>${esc(s.name)}</td><td class="num">${s.imbalance_ratio === null ? "-" : s.imbalance_ratio + "×"}</td>
        <td>${s.absent_labels.length ? esc(s.absent_labels.join(", ")) : "none"}</td><td class="num">${num(empty)}</td>
        <td class="num">${s.answer_words_mean} / ${s.answer_words_p95}</td></tr>`;
    }).join("")}</tbody></table></div>
    <p class="help mt-sm">Non-Answer is tiny (well under 1% of answers), which is why the models report macro-F1 alongside weighted-F1
    and why training used class-weighted loss. Unseen Questions contains no Non-Answer examples at all.</p>`;
}

async function samples(el, splits, f) {
  const labels = [["", "All labels"], ["correct", "Correct"], ["partially_correct_incomplete", "Partially Correct"],
    ["contradictory", "Contradictory"], ["irrelevant", "Irrelevant"], ["non_domain", "Non-Answer"]];
  el.innerHTML = `<div class="card-head"><div><h2>Browse answers</h2><p>Real examples from the public dataset.</p></div></div>
    <div class="controls" style="margin-bottom:12px">
      <select id="s-split" aria-label="Split">${splits.map((s) => `<option value="${s.key}" ${s.key === f.split ? "selected" : ""}>${esc(s.name)}</option>`).join("")}</select>
      <select id="s-label" aria-label="Label">${labels.map(([k, n]) => `<option value="${k}" ${k === f.label ? "selected" : ""}>${n}</option>`).join("")}</select>
      <input id="s-q" type="search" placeholder="Search text…" value="${esc(f.q)}" style="width:240px" aria-label="Search answers">
    </div><div id="s-body">${loading()}</div>`;
  const reload = (change) => samples(el, splits, { ...f, ...change, offset: change.offset ?? 0 });
  $("#s-split", el).addEventListener("change", (e) => reload({ split: e.target.value }));
  $("#s-label", el).addEventListener("change", (e) => reload({ label: e.target.value }));
  $("#s-q", el).addEventListener("input", debounce((e) => reload({ q: e.target.value }), 350));
  if (f.q) { const q = $("#s-q", el); q.focus(); q.setSelectionRange(q.value.length, q.value.length); }

  let d;
  try {
    d = await api("/api/dataset/samples", { params: { split: f.split, label: f.label, q: f.q, offset: f.offset, limit: 20 } });
  } catch (e) {
    $("#s-body", el).innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  if (!d.total) {
    $("#s-body", el).innerHTML = emptyState("No matching answers", "Try another label or search term.", "search");
    refreshIcons();
    return;
  }
  $("#s-body", el).innerHTML = `<div class="table-wrap"><table><thead><tr><th>Question</th><th>Reference answer</th><th>Student answer</th><th>Label</th></tr></thead>
    <tbody>${d.items.map((r) => `<tr><td class="cell-text">${esc(r.question)}</td><td class="cell-text">${esc(r.reference_answer)}</td>
      <td class="cell-text">${esc(r.student_answer)}</td><td>${labelBadge(r.label)}</td></tr>`).join("")}</tbody></table></div>${pager(d.total, f.offset, 20)}`;
  $$("[data-page]", el).forEach((b) => b.addEventListener("click", () =>
    reload({ offset: Math.max(0, f.offset + (b.dataset.page === "next" ? 20 : -20)) })));
}

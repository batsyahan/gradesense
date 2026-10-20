import { api, download } from "../api.js";
import { simpleBar } from "../charts.js";
import { $, $$, alertBox, debounce, emptyState, esc, icon, labelBadge, LABEL_NAME, loading, num, pager, pct, refreshIcons } from "../ui.js";

const MODELS = [["DeBERTa-v3", "DeBERTa-v3 (FT)"], ["SBERT-LR", "SBERT (No FT) + LR"], ["BiLSTM", "Siamese BiLSTM"],
  ["TFIDF-Full", "TF-IDF + LR"], ["TFIDF-Sim", "TF-IDF similarity + LR"]];
const SPLITS = [["test_ua", "Unseen Answers"], ["test_uq", "Unseen Questions"], ["test_ud", "Unseen Domains"], ["val", "Validation"]];
const LABELS = ["correct", "partially_correct_incomplete", "contradictory", "irrelevant", "non_domain"];

let filters = { model: "DeBERTa-v3", split: "test_ua", true: "", pred: "", q: "", offset: 0 };

export function render(view) {
  view.innerHTML = `
    <div class="page-head"><div><h1>Error Analysis</h1>
      <p>Where each model goes wrong on held-out answers, from its saved predictions.</p></div>
      <button class="btn" id="e-export">${icon("download")}Export CSV</button></div>
    <section class="card card-pad">
      <div class="controls">
        <select id="e-model" aria-label="Model">${MODELS.map(([k, n]) => `<option value="${k}" ${k === filters.model ? "selected" : ""}>${n}</option>`).join("")}</select>
        <select id="e-split" aria-label="Split">${SPLITS.map(([k, n]) => `<option value="${k}" ${k === filters.split ? "selected" : ""}>${n}</option>`).join("")}</select>
        <select id="e-true" aria-label="True label"><option value="">Any true label</option>${LABELS.map((l) => `<option value="${l}" ${l === filters.true ? "selected" : ""}>True: ${LABEL_NAME[l]}</option>`).join("")}</select>
        <select id="e-pred" aria-label="Predicted label"><option value="">Any prediction</option>${LABELS.map((l) => `<option value="${l}" ${l === filters.pred ? "selected" : ""}>Predicted: ${LABEL_NAME[l]}</option>`).join("")}</select>
        <input id="e-q" type="search" placeholder="Search answers…" value="${esc(filters.q)}" style="width:220px" aria-label="Search">
      </div>
    </section>
    <div id="e-content" class="mt">${loading()}</div>`;

  const set = (change) => { filters = { ...filters, ...change, offset: change.offset ?? 0 }; load(view); };
  $("#e-model", view).addEventListener("change", (e) => set({ model: e.target.value }));
  $("#e-split", view).addEventListener("change", (e) => set({ split: e.target.value }));
  $("#e-true", view).addEventListener("change", (e) => set({ true: e.target.value }));
  $("#e-pred", view).addEventListener("change", (e) => set({ pred: e.target.value }));
  $("#e-q", view).addEventListener("input", debounce((e) => set({ q: e.target.value }), 350));
  $("#e-export", view).addEventListener("click", async () => {
    try {
      await download("/api/errors/export", { model: filters.model, split: filters.split, true: filters.true, pred: filters.pred, q: filters.q },
        `errors_${filters.model}_${filters.split}.csv`);
    } catch (e) { alert(e.message); }
  });
  refreshIcons();
  load(view);
}

async function load(view) {
  const box = $("#e-content", view);
  if (!box) return;
  box.innerHTML = loading();
  let d;
  try {
    d = await api("/api/errors", { params: { model: filters.model, split: filters.split, true: filters.true, pred: filters.pred, q: filters.q, offset: filters.offset, limit: 25 } });
  } catch (e) {
    box.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  if (!d.available) {
    box.innerHTML = `<section class="card card-pad">${emptyState("No predictions for this model and split", esc(d.message), "file-search")}</section>`;
    refreshIcons();
    return;
  }
  const s = d.summary;
  box.innerHTML = `
    <div class="grid grid-3">
      <div class="card stat"><div class="stat-body"><div class="stat-label">Misclassified</div><div class="stat-value sm">${num(s.errors)} <span class="help">of ${num(s.total)}</span></div><div class="stat-note">error rate ${pct(s.error_rate, 1)}</div></div></div>
      <div class="card stat"><div class="stat-body"><div class="stat-label">Most common error</div><div class="stat-value sm" style="font-size:16px">${s.categories[0] ? esc(s.categories[0].category) : "-"}</div><div class="stat-note">${s.categories[0] ? num(s.categories[0].count) + " answers" : ""}</div></div></div>
      <div class="card stat"><div class="stat-body"><div class="stat-label">Most common confusion</div><div class="stat-value sm" style="font-size:16px">${s.pairs[0] ? `${esc(LABEL_NAME[s.pairs[0].true])} → ${esc(LABEL_NAME[s.pairs[0].pred])}` : "-"}</div><div class="stat-note">${s.pairs[0] ? num(s.pairs[0].count) + " answers (true → predicted)" : ""}</div></div></div>
    </div>
    <div class="grid grid-2 mt">
      <section class="card card-pad"><h2>Error categories</h2><div class="chart-box short mt-sm"><canvas role="img" aria-label="Number of errors per category"></canvas></div>
        <p class="help mt-sm">Categories are assigned by fixed rules on (true, predicted) label pairs, e.g. a correct answer predicted as anything else is "under-credited".</p></section>
      <section class="card card-pad"><h2>False positives and false negatives</h2>
        <div class="table-wrap mt-sm"><table><thead><tr><th>Label</th><th class="num">Support</th><th class="num">False negatives</th><th class="num">False positives</th></tr></thead>
        <tbody>${s.per_class.map((c) => `<tr class="clickable" data-true="${c.label}"><td>${labelBadge(c.label)}</td><td class="num">${num(c.support)}</td>
          <td class="num">${num(c.false_negatives)}</td><td class="num">${num(c.false_positives)}</td></tr>`).join("")}</tbody></table></div>
        <p class="help mt-sm">False negative: the answer had this label but the model predicted another. False positive: the model predicted this label wrongly. Select a row to filter.</p></section>
    </div>
    <section class="card card-pad mt"><div class="card-head"><div><h2>Misclassified answers</h2><p>${num(d.total)} matching the filters</p></div></div>
      ${d.items.length ? `<div class="table-wrap"><table><thead><tr><th>Question</th><th>Reference answer</th><th>Student answer</th><th>True</th><th>Predicted</th><th>Category</th></tr></thead>
      <tbody>${d.items.map((r) => `<tr><td class="cell-text">${esc(r.question)}</td><td class="cell-text">${esc(r.reference_answer)}</td><td class="cell-text"><strong>${esc(r.student_answer)}</strong></td>
        <td>${labelBadge(r.true_label)}</td><td>${labelBadge(r.pred_label)}</td><td>${esc(r.category)}</td></tr>`).join("")}</tbody></table></div>${pager(d.total, filters.offset, 25)}`
      : emptyState("No errors match these filters", "Try clearing a filter.", "circle-check")}
    </section>`;

  simpleBar($("canvas", box), {
    labels: s.categories.map((c) => c.category.split(":")[0]), data: s.categories.map((c) => c.count), yTitle: "Errors",
    colors: "#3b82f6",
  });
  $$("tr[data-true]", box).forEach((tr) => tr.addEventListener("click", () => {
    filters = { ...filters, true: tr.dataset.true, pred: "", offset: 0 };
    $("#e-true", view).value = filters.true;
    $("#e-pred", view).value = "";
    load(view);
  }));
  $$("[data-page]", box).forEach((b) => b.addEventListener("click", () => {
    filters.offset = Math.max(0, filters.offset + (b.dataset.page === "next" ? 25 : -25));
    load(view);
  }));
  refreshIcons();
}

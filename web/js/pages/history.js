import { api, download } from "../api.js";
import { state } from "../state.js";
import { $, $$, alertBox, debounce, emptyState, esc, icon, loading, marks, num, pager, pct, refreshIcons, toast, verdictBadge } from "../ui.js";

let f = { q: "", offset: 0 };

export function render(view, params) {
  f = { q: params.q || "", offset: 0 };
  view.innerHTML = `
    <div class="page-head"><div><h1>Grading History</h1>
      <p>Every answer graded from this browser - single answers and batch rows.</p></div>
      <div class="btn-row"><button class="btn" id="h-export">${icon("download")}Export CSV</button>
      <button class="btn danger" id="h-clear">${icon("trash-2")}Delete my history</button></div></div>
    <section class="card card-pad">
      ${alertBox("info", "History is linked to a random ID stored in this browser, so only you see these records. They are stored in the server's database; on free hosting the database is reset whenever the server restarts.")}
      <div class="controls mt"><input id="h-q" type="search" placeholder="Search questions, answers or verdicts…" value="${esc(f.q)}" style="width:320px" aria-label="Search history"></div>
      <div id="h-body" class="mt">${loading()}</div>
    </section>`;
  $("#h-q", view).addEventListener("input", debounce((e) => { f = { q: e.target.value, offset: 0 }; load(view); }, 300));
  $("#h-export", view).addEventListener("click", async () => {
    try { await download("/api/history/export", null, "grading_history.csv"); } catch (e) { alert(e.message); }
  });
  $("#h-clear", view).addEventListener("click", async () => {
    if (!confirm("Delete all grading records made from this browser? This cannot be undone.")) return;
    try {
      const r = await api("/api/history", { method: "DELETE" });
      toast(`Deleted ${num(r.deleted)} records.`);
      load(view);
    } catch (e) { alert(e.message); }
  });
  refreshIcons();
  load(view);
}

async function load(view) {
  const body = $("#h-body", view);
  if (!body) return;
  let d;
  try {
    d = await api("/api/history", { params: { q: f.q, offset: f.offset, limit: 25 } });
  } catch (e) {
    body.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  if (!d.enabled) {
    body.innerHTML = emptyState("History is turned off", esc(d.error || "This server does not store grading records."), "history");
    refreshIcons();
    return;
  }
  if (!d.total) {
    body.innerHTML = emptyState(f.q ? "No matching records" : "No answers graded yet",
      f.q ? "Try a different search." : `Grade an answer on the <a href="#/grade">Grade One</a> page and it will appear here.`, "history");
    refreshIcons();
    return;
  }
  body.innerHTML = `<div class="table-wrap"><table><thead><tr><th>When (UTC)</th><th>Source</th><th>Question</th><th>Student answer</th><th>Verdict</th><th class="num">Marks</th><th class="num">Confidence</th><th></th></tr></thead>
    <tbody>${d.items.map((r) => `<tr><td style="white-space:nowrap">${esc(r.created_at.replace("T", " ").slice(0, 16))}</td><td>${esc(r.source)}</td>
      <td class="cell-text">${esc(r.question)}</td><td class="cell-text">${esc(r.answer)}</td><td>${verdictBadge(r.verdict)}</td>
      <td class="num">${marks(r.marks)} / ${marks(r.max_marks)}</td><td class="num">${pct(r.confidence)}</td>
      <td><button class="btn sm" data-open="${r.id}">Open</button></td></tr>`).join("")}</tbody></table></div>${pager(d.total, f.offset, 25)}`;
  $$("[data-open]", body).forEach((b) => b.addEventListener("click", () => {
    const r = d.items.find((x) => String(x.id) === b.dataset.open);
    state.draft = { question: r.question, reference: r.reference, answer: r.answer, max_marks: r.max_marks };
    state.lastResult = null;
    location.hash = "#/grade";
  }));
  $$("[data-page]", body).forEach((b) => b.addEventListener("click", () => {
    f.offset = Math.max(0, f.offset + (b.dataset.page === "next" ? 25 : -25));
    load(view);
  }));
  refreshIcons();
}

// Grading workspace: the "Grade a Single Answer" form and the "Grading Result" panel.
import { api } from "../api.js";
import { effectiveMapping, prefs, setLastResult, state } from "../state.js";
import {
  $, $$, alertBox, emptyState, esc, icon, marks, pct, probRows, refreshIcons, verdictBadge, VERDICT_COLOR,
} from "../ui.js";

const FIELDS = [
  { key: "question", label: "Question", limit: "question", rows: 2 },
  { key: "reference", label: "Reference Answer", limit: "reference", rows: 3 },
  { key: "answer", label: "Student Answer", limit: "answer", rows: 3 },
];

export function renderWorkspace(root, { showOptions = false } = {}) {
  const meta = state.meta;
  const p = prefs();
  const examples = meta.examples || [];
  const draft = state.draft || (state.lastResult && state.lastResult.inputs) || {};
  state.draft = null;

  root.innerHTML = `
  <div class="grid grid-ws">
    <section class="card card-pad" aria-labelledby="ws-title">
      <div class="card-head">
        <div><h2 id="ws-title">Grade a Single Answer</h2>
        <p>Enter a question, reference answer and student answer to get AI grading and feedback.</p></div>
        <div>
          <label for="ws-example" class="sr-only">Load an example</label>
          <select id="ws-example" class="btn sm" style="width:auto">
            <option value="">Load example…</option>
            ${examples.map((e, i) => `<option value="${i}">${esc(e.name)}</option>`).join("")}
          </select>
        </div>
      </div>
      <form id="ws-form" novalidate>
        ${FIELDS.map((f) => `
          <div class="field-wrap">
            <label class="field" for="ws-${f.key}">${f.label}</label>
            <textarea id="ws-${f.key}" name="${f.key}" rows="${f.rows}" maxlength="${meta.limits[f.limit]}"
              aria-describedby="ws-${f.key}-count">${esc(draft[f.key] || "")}</textarea>
            <span class="counter" id="ws-${f.key}-count" aria-live="off"></span>
          </div>`).join("")}
        <div class="inline" style="margin-bottom:14px">
          <label for="ws-max">Maximum marks</label>
          <input id="ws-max" type="number" min="0.5" max="100" step="0.5" value="${esc(draft.max_marks || p.defaultMaxMarks)}">
          ${showOptions ? `<label class="check"><input id="ws-explain" type="checkbox" ${p.explain ? "checked" : ""}>Word-level highlighting</label>` : ""}
        </div>
        <div id="ws-error" class="field-error" role="alert"></div>
        <div class="btn-row" style="margin-top:6px">
          <button type="submit" class="btn primary" id="ws-submit">${icon("play")}Grade Answer</button>
          <button type="button" class="btn" id="ws-clear">Clear All</button>
        </div>
      </form>
    </section>
    <section class="card card-pad" id="ws-result" aria-live="polite" aria-labelledby="res-title"></section>
  </div>`;

  const form = $("#ws-form", root);
  const err = $("#ws-error", root);
  const submit = $("#ws-submit", root);

  function updateCount(f) {
    const ta = $(`#ws-${f.key}`, root);
    const c = $(`#ws-${f.key}-count`, root);
    const limit = meta.limits[f.limit];
    c.textContent = `${ta.value.length}/${limit}`;
    c.classList.toggle("over", ta.value.length > limit);
  }
  FIELDS.forEach((f) => {
    updateCount(f);
    $(`#ws-${f.key}`, root).addEventListener("input", () => updateCount(f));
  });

  $("#ws-example", root).addEventListener("change", (e) => {
    const ex = examples[Number(e.target.value)];
    if (!ex) return;
    $("#ws-question", root).value = ex.question;
    $("#ws-reference", root).value = ex.reference;
    $("#ws-answer", root).value = ex.answer;
    FIELDS.forEach(updateCount);
    err.textContent = "";
  });

  $("#ws-clear", root).addEventListener("click", () => {
    FIELDS.forEach((f) => { $(`#ws-${f.key}`, root).value = ""; updateCount(f); });
    $("#ws-example", root).value = "";
    err.textContent = "";
    setLastResult(null);
    renderResult($("#ws-result", root), null);
  });

  let running = false;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (running) return;
    const inputs = {
      question: $("#ws-question", root).value.trim(),
      reference: $("#ws-reference", root).value.trim(),
      answer: $("#ws-answer", root).value.trim(),
      max_marks: Number($("#ws-max", root).value),
    };
    const problems = [];
    FIELDS.forEach((f) => {
      if (!inputs[f.key]) problems.push(`${f.label} is required.`);
      else if (inputs[f.key].length > meta.limits[f.limit]) problems.push(`${f.label} is longer than ${meta.limits[f.limit]} characters.`);
    });
    if (!(inputs.max_marks > 0 && inputs.max_marks <= 100)) problems.push("Maximum marks must be between 0 and 100.");
    if (problems.length) { err.textContent = problems.join(" "); return; }
    err.textContent = "";

    const explainBox = $("#ws-explain", root);
    const explain = explainBox ? explainBox.checked : p.explain;
    running = true;
    submit.disabled = true;
    submit.innerHTML = `<span class="spinner"></span>Grading…`;
    renderResult($("#ws-result", root), "loading");
    try {
      const result = await api("/api/grade", {
        method: "POST",
        body: { ...inputs, explain, marks_mapping: effectiveMapping() },
      });
      setLastResult({ inputs, result });
      renderResult($("#ws-result", root), result);
    } catch (ex) {
      renderResult($("#ws-result", root), { error: ex.message, status: ex.status });
    } finally {
      running = false;
      submit.disabled = false;
      submit.innerHTML = `${icon("play")}Grade Answer`;
      refreshIcons();
    }
  });

  renderResult($("#ws-result", root), state.lastResult ? state.lastResult.result : null);
}

export function renderResult(el, result) {
  const head = (badge = "") => `<div class="card-head"><h2 id="res-title">Grading Result</h2>${badge}</div>`;
  if (result === "loading") {
    el.innerHTML = head() + `<div class="loading-block"><span class="spinner dark"></span>Running the model…</div>`;
    return;
  }
  if (!result) {
    el.innerHTML = head() + emptyState("No answer graded yet",
      "Fill in the form or load an example, then select <strong>Grade Answer</strong>. Results appear here.", "clipboard-check");
    refreshIcons();
    return;
  }
  if (result.error) {
    el.innerHTML = head() + alertBox(result.status === 503 ? "warn" : "error", esc(result.error));
    refreshIcons();
    return;
  }

  const p = prefs();
  const color = VERDICT_COLOR[result.verdict] || "gray";
  const confColor = result.confidence >= 0.8 ? "" : result.confidence >= 0.6 ? "amber" : "red";
  const fb = result.feedback;
  const section = (title, items) => items && items.length
    ? `<div class="fb-section"><h4>${title}</h4><ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul></div>` : "";

  el.innerHTML = head(verdictBadge(result.verdict)) + `
    <div class="result-top">
      <div class="marks-box ${color}">
        <div class="marks-value">${marks(result.marks)} <span>/ ${marks(result.max_marks)}</span></div>
        <div class="marks-label">Marks awarded</div>
      </div>
      ${p.showConfidence ? `
      <div class="conf-box">
        <div class="conf-head"><span>Model confidence</span><span>${result.confidence.toFixed(2)}</span></div>
        <div class="bar ${confColor}"><div style="width:${(result.confidence * 100).toFixed(1)}%"></div></div>
        <div class="meta-line">Softmax probability of the predicted label - not a guarantee of correctness.</div>
      </div>` : `<div class="conf-box"><div class="meta-line">Confidence display is turned off in Model Configuration.</div></div>`}
    </div>
    <div class="meta-line">Model label (5-class): <strong>${esc(result.label_name)}</strong> · counts as <strong>${esc(result.verdict)}</strong>${result.saved_to_history ? " · saved to your history" : ""}</div>
    ${result.needs_review ? `<div class="mt-sm">${alertBox("warn", "Low model confidence - a teacher should review this grade.")}</div>` : ""}
    <h3 class="mt" style="margin-bottom:4px">Feedback &amp; Analysis</h3>
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" id="tab-c" aria-controls="panel-c" aria-selected="true">Concept Coverage</button>
      <button class="tab" role="tab" id="tab-h" aria-controls="panel-h" aria-selected="false" tabindex="-1">Highlighted Answer</button>
      <button class="tab" role="tab" id="tab-f" aria-controls="panel-f" aria-selected="false" tabindex="-1">Detailed Feedback</button>
    </div>
    <div class="tab-panel" id="panel-c" role="tabpanel" aria-labelledby="tab-c">${conceptsHtml(result.concepts)}</div>
    <div class="tab-panel" id="panel-h" role="tabpanel" aria-labelledby="tab-h" hidden>${highlightHtml(result.highlights)}</div>
    <div class="tab-panel" id="panel-f" role="tabpanel" aria-labelledby="tab-f" hidden>
      <p style="margin-bottom:12px">${esc(fb.summary)}</p>
      ${section("What the answer got right", fb.got_right)}
      ${section("What is missing", fb.missing)}
      ${section("What is incorrect", fb.incorrect)}
      ${section("How to improve", fb.improve)}
      <div class="fb-section"><h4>Model probabilities (5 classes)</h4>${probRows(result.probabilities)}</div>
    </div>`;

  const tabs = $$(".tab", el);
  function select(tab) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $("#" + t.getAttribute("aria-controls"), el).hidden = !on;
    });
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        select(next);
        next.focus();
      }
    });
  });
  refreshIcons();
}

function conceptsHtml(concepts) {
  if (!concepts || !concepts.length) {
    return emptyState("No key concepts found", "Concepts could not be extracted from this reference answer.", "list");
  }
  const look = {
    "Covered": ["circle-check", "ic-green", "green"],
    "Partially Covered": ["circle-alert", "ic-amber", "amber"],
    "Missing": ["circle-x", "ic-red", "red"],
  };
  const rows = concepts.map((c) => {
    const [ic, icCls, badge] = look[c.status];
    const detail = c.method === "exact" ? "same wording found in the answer"
      : c.match ? `closest phrase: “${esc(c.match)}” (similarity ${c.score.toFixed(2)})`
      : c.method === "semantic" ? `best similarity ${c.score.toFixed(2)}` : "not found";
    return `<div class="concept-row">${icon(ic, icCls)}<div class="name">${esc(c.concept)}<small>${detail}${c.note ? " · " + esc(c.note) : ""}</small></div>
      <span class="badge ${badge}">${esc(c.status)}</span></div>`;
  }).join("");
  return `<div style="font-weight:600;margin-bottom:8px">Key Concepts</div>${rows}
    <p class="help mt-sm">Concepts come from the reference answer (spaCy noun phrases and content words). Coverage uses exact word
    matches and Sentence-BERT similarity - a heuristic, not proof that the student understood the idea.</p>`;
}

function highlightHtml(h) {
  if (!h || !h.tokens || !h.tokens.length) return emptyState("No highlights", "Nothing to highlight for this answer.", "highlighter");
  const tokens = h.tokens.map((t) => {
    const title = t.status === "neutral" ? "" :
      ` title="${t.status}${h.occlusion ? `: removing this word changes the expected marks by ${(t.delta * -100).toFixed(1)} percentage points` : ""}"`;
    return t.status === "neutral" ? esc(t.text) : `<span class="tok ${t.status}"${title}>${esc(t.text)}</span>`;
  }).join(" ");
  return `<div class="answer-hl">${tokens}</div>
    <div class="legend"><span><i style="background:#dcfce7;border:1px solid #16a34a"></i>Supports the grade / matches a concept</span>
    <span><i style="background:#fef3c7;border:1px solid #d97706"></i>Partially matches a concept</span>
    <span><i style="background:#fee2e2;border:1px solid #dc2626"></i>Lowered the score</span></div>
    <p class="help mt-sm">${esc(h.method)}</p>`;
}

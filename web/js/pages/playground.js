import { api } from "../api.js";
import { state } from "../state.js";
import { $, alertBox, esc, icon, probRows, refreshIcons, verdictBadge } from "../ui.js";

export function render(view) {
  const ex = state.meta.examples;
  view.innerHTML = `
    <div class="page-head"><div><h1>Model Playground</h1>
      <p>Grade one answer with every model side by side and see where the simpler models fail.</p></div></div>
    <section class="card card-pad">
      <div class="card-head"><div><h2>Answer to compare</h2><p>Tip: try the contradiction and paraphrase examples - they separate the models most clearly.</p></div>
        <select id="pg-ex" class="btn sm" style="width:auto" aria-label="Load an example"><option value="">Load example…</option>${ex.map((e, i) => `<option value="${i}">${esc(e.name)}</option>`).join("")}</select></div>
      <div class="grid grid-3">
        <div><label class="field" for="pg-q">Question</label><textarea id="pg-q" rows="3" maxlength="${state.meta.limits.question}"></textarea></div>
        <div><label class="field" for="pg-r">Reference Answer</label><textarea id="pg-r" rows="3" maxlength="${state.meta.limits.reference}"></textarea></div>
        <div><label class="field" for="pg-a">Student Answer</label><textarea id="pg-a" rows="3" maxlength="${state.meta.limits.answer}"></textarea></div>
      </div>
      <div id="pg-err" class="field-error" role="alert"></div>
      <div class="btn-row mt-sm"><button class="btn primary" id="pg-run">${icon("flask-conical")}Compare models</button>
        <button class="btn" id="pg-grade">Open in Grade One</button></div>
    </section>
    <div id="pg-out" class="mt" aria-live="polite"></div>`;

  const fill = (e) => { $("#pg-q", view).value = e.question; $("#pg-r", view).value = e.reference; $("#pg-a", view).value = e.answer; };
  fill(ex.find((e) => e.key === "paraphrase") || ex[0]);
  $("#pg-ex", view).addEventListener("change", (e) => { if (e.target.value !== "") fill(ex[Number(e.target.value)]); });
  const inputs = () => ({ question: $("#pg-q", view).value.trim(), reference: $("#pg-r", view).value.trim(), answer: $("#pg-a", view).value.trim() });

  $("#pg-grade", view).addEventListener("click", () => { state.draft = inputs(); state.lastResult = null; location.hash = "#/grade"; });
  $("#pg-run", view).addEventListener("click", async () => {
    const body = inputs();
    const err = $("#pg-err", view);
    if (!body.question || !body.reference || !body.answer) { err.textContent = "Question, reference answer and student answer are all required."; return; }
    err.textContent = "";
    const btn = $("#pg-run", view);
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Running 4 models…`;
    try {
      const d = await api("/api/playground", { method: "POST", body });
      renderModels($("#pg-out", view), d.models);
    } catch (e) {
      $("#pg-out", view).innerHTML = alertBox(e.status === 503 ? "warn" : "error", esc(e.message));
    } finally {
      btn.disabled = false;
      btn.innerHTML = `${icon("flask-conical")}Compare models`;
      refreshIcons();
    }
  });
  refreshIcons();
}

function renderModels(el, models) {
  const verdicts = new Set(models.filter((m) => m.available).map((m) => m.label));
  el.innerHTML = `${verdicts.size > 1 ? alertBox("info", "The models disagree on this answer. The fine-tuned DeBERTa-v3 is the model GradeSense uses for grading.") : ""}
    <div class="grid grid-4 mt-sm">${models.map((m) => `
      <section class="card pg-card ${m.key === "DeBERTa-v3" ? "hero" : ""}">
        <h3>${esc(m.name)}</h3><p class="help" style="min-height:38px">${esc(m.description)}</p>
        ${m.available ? `<div class="mt-sm">${verdictBadge(m.verdict)}</div>
          <div class="meta-line">5-class label: <strong>${esc(m.label_name)}</strong> · ${(m.confidence * 100).toFixed(0)}% · ${m.ms} ms</div>
          <div class="mt-sm">${probRows(m.probabilities)}</div>`
        : `<div class="mt-sm">${alertBox("warn", esc(m.error || "Not available"))}</div>`}
      </section>`).join("")}</div>
    <p class="help mt-sm">Each model outputs probabilities for the same five labels. Baselines were trained on the same SciEntsBank training split; see Model Insights for their test scores.</p>`;
  refreshIcons();
}

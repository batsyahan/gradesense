import { api } from "../api.js";
import { LABEL_KEYS, prefs, resetPrefs, savePrefs, state } from "../state.js";
import { $, $$, alertBox, esc, icon, loading, refreshIcons, toast } from "../ui.js";

const LABEL_TITLES = {
  correct: "Correct", partially_correct_incomplete: "Partially Correct", contradictory: "Contradictory",
  irrelevant: "Irrelevant", non_domain: "Non-Answer",
};
const GROUPS = [
  ["Model", ["model_dir", "model_repo", "device", "max_length", "batch_size"]],
  ["Grading", ["marks_mapping", "review_threshold", "concept_covered_threshold", "concept_partial_threshold", "occlusion_max_words", "show_confidence"]],
  ["Artifacts and storage", ["results_dir", "history_enabled", "db_path"]],
  ["Limits", ["max_question_chars", "max_reference_chars", "max_answer_chars", "max_batch_rows", "max_upload_mb"]],
  ["Display", ["profile_name", "profile_subtitle", "allow_config_edit"]],
];

export async function render(view) {
  view.innerHTML = `
    <div class="page-head"><div><h1>Model Configuration</h1>
      <p>Inference and display settings. Changing them never retrains the model.</p></div></div>
    <section class="card card-pad" id="c-prefs"></section>
    <section class="card card-pad mt" id="c-server">${loading()}</section>`;
  renderPrefs($("#c-prefs", view));
  renderServer($("#c-server", view));
}

function renderPrefs(el) {
  const p = prefs();
  const mapping = p.marksMapping || state.meta.marks_mapping;
  el.innerHTML = `<div class="card-head"><div><h2>Your preferences</h2><p>Saved in this browser only. They change how results are shown and how marks are calculated for your gradings.</p></div></div>
    <div class="grid grid-2">
      <div>
        <h3>Marks mapping</h3><p class="help">Share of maximum marks awarded for each model label (0 to 1).</p>
        <div class="table-wrap mt-sm"><table><tbody>${LABEL_KEYS.map((k) => `<tr><td><label for="mm-${k}">${LABEL_TITLES[k]}</label></td>
          <td class="num"><input id="mm-${k}" type="number" min="0" max="1" step="0.05" value="${mapping[k]}" style="width:90px"></td></tr>`).join("")}</tbody></table></div>
      </div>
      <div>
        <h3>Defaults</h3>
        <div class="field-wrap mt-sm"><label class="field" for="pf-max">Default maximum marks</label>
          <input id="pf-max" type="number" min="0.5" max="100" step="0.5" value="${p.defaultMaxMarks}" style="width:120px"></div>
        <div class="field-wrap"><label class="check"><input type="checkbox" id="pf-explain" ${p.explain ? "checked" : ""}>Word-level highlighting by default (adds a few seconds per answer)</label></div>
        <div class="field-wrap"><label class="check"><input type="checkbox" id="pf-conf" ${p.showConfidence ? "checked" : ""}>Show model confidence in results</label>
          <p class="help">Confidence is the softmax probability of the predicted label. It is not calibrated and is not a guarantee of correctness.</p></div>
      </div>
    </div>
    <div id="pf-err" class="field-error" role="alert"></div>
    <div class="btn-row mt"><button class="btn primary" id="pf-save">Save preferences</button><button class="btn" id="pf-reset">Reset to server defaults</button></div>`;

  $("#pf-save", el).addEventListener("click", () => {
    const mm = {};
    for (const k of LABEL_KEYS) {
      const v = Number($(`#mm-${k}`, el).value);
      if (!(v >= 0 && v <= 1)) { $("#pf-err", el).textContent = `${LABEL_TITLES[k]}: enter a value between 0 and 1.`; return; }
      mm[k] = v;
    }
    const max = Number($("#pf-max", el).value);
    if (!(max > 0 && max <= 100)) { $("#pf-err", el).textContent = "Default maximum marks must be between 0 and 100."; return; }
    $("#pf-err", el).textContent = "";
    savePrefs({ marksMapping: mm, defaultMaxMarks: max, explain: $("#pf-explain", el).checked, showConfidence: $("#pf-conf", el).checked });
    toast("Preferences saved in this browser.");
  });
  $("#pf-reset", el).addEventListener("click", () => { resetPrefs(); renderPrefs(el); toast("Preferences reset."); });
}

async function renderServer(el) {
  let d;
  try { d = await api("/api/config"); }
  catch (e) { el.innerHTML = alertBox("error", esc(e.message)); refreshIcons(); return; }
  const byKey = Object.fromEntries(d.settings.map((s) => [s.key, s]));
  const editable = d.editable;

  const input = (s) => {
    const dis = !editable || s.locked ? "disabled" : "";
    if (s.key === "marks_mapping") {
      return LABEL_KEYS.map((k) => `<div class="inline" style="margin-bottom:4px"><label for="sv-mm-${k}" style="width:140px">${LABEL_TITLES[k]}</label>
        <input id="sv-mm-${k}" type="number" min="0" max="1" step="0.05" value="${s.value[k]}" ${dis}></div>`).join("");
    }
    if (s.type === "bool") return `<label class="check"><input type="checkbox" id="sv-${s.key}" ${s.value ? "checked" : ""} ${dis}>Enabled</label>`;
    if (s.key === "device") return `<select id="sv-${s.key}" ${dis} style="width:auto"><option value="cpu" ${s.value === "cpu" ? "selected" : ""}>cpu</option><option value="cuda" ${s.value === "cuda" ? "selected" : ""}>cuda (GPU)</option></select>`;
    if (s.type === "int" || s.type === "float") {
      return `<input id="sv-${s.key}" type="number" value="${s.value}" ${s.min !== null ? `min="${s.min}"` : ""} ${s.max !== null ? `max="${s.max}"` : ""} step="${s.type === "int" ? 1 : 0.01}" ${dis} style="width:140px">`;
    }
    return `<input id="sv-${s.key}" type="text" value="${esc(s.value)}" ${dis}>`;
  };

  el.innerHTML = `<div class="card-head"><div><h2>Server settings</h2><p>Apply to everyone using this server.</p></div>
      ${editable ? `<button class="btn" id="sv-reload">${icon("refresh-cw")}Reload model</button>` : ""}</div>
    ${editable ? alertBox("info", "Changing model or inference settings reloads the existing model files. Nothing is retrained.")
      : alertBox("info", "Settings are read-only on this server. To change them, run GradeSense locally with <code>\"allow_config_edit\": true</code> in <code>config.json</code>.")}
    ${GROUPS.map(([title, keys]) => `<h3 class="mt">${title}</h3><div class="table-wrap mt-sm"><table><tbody>
      ${keys.filter((k) => byKey[k]).map((k) => { const s = byKey[k]; return `<tr><td style="width:38%"><strong>${esc(k)}</strong><br><small class="help">${esc(s.description)}${s.locked ? " · fixed by the server" : ""}</small></td>
        <td>${input(s)}<div class="field-error" id="sv-err-${k}"></div></td></tr>`; }).join("")}</tbody></table></div>`).join("")}
    ${editable ? `<div id="sv-msg" class="mt"></div><div class="btn-row mt"><button class="btn primary" id="sv-save">Save server settings</button></div>` : ""}`;

  if (!editable) { refreshIcons(); return; }
  $("#sv-reload", el).addEventListener("click", async () => {
    try { await api("/api/model/reload", { method: "POST" }); toast("Model reloading - this takes about a minute."); }
    catch (e) { alert(e.message); }
  });
  $("#sv-save", el).addEventListener("click", async () => {
    const updates = {};
    d.settings.filter((s) => !s.locked).forEach((s) => {
      if (s.key === "marks_mapping") {
        updates[s.key] = Object.fromEntries(LABEL_KEYS.map((k) => [k, Number($(`#sv-mm-${k}`, el).value)]));
      } else if (s.type === "bool") {
        updates[s.key] = $(`#sv-${s.key}`, el).checked;
      } else if (s.type === "int" || s.type === "float") {
        updates[s.key] = Number($(`#sv-${s.key}`, el).value);
      } else {
        updates[s.key] = $(`#sv-${s.key}`, el).value;
      }
      if (JSON.stringify(updates[s.key]) === JSON.stringify(s.value)) delete updates[s.key];
    });
    $$("[id^=sv-err-]", el).forEach((x) => { x.textContent = ""; });
    const msg = $("#sv-msg", el);
    if (!Object.keys(updates).length) { msg.innerHTML = alertBox("info", "No changes to save."); refreshIcons(); return; }
    try {
      const r = await api("/api/config", { method: "PUT", body: { updates } });
      msg.innerHTML = alertBox("success", `Saved: ${esc(r.saved.join(", "))}.${r.model_reloading ? " The model is reloading with the new settings (not retrained)." : ""}`);
    } catch (e) {
      if (e.details) Object.entries(e.details).forEach(([k, v]) => { const x = $(`#sv-err-${k}`, el); if (x) x.textContent = v; });
      msg.innerHTML = alertBox("error", esc(e.message));
    }
    refreshIcons();
  });
  refreshIcons();
}

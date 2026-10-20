import { api, download, saveBlob } from "../api.js";
import { simpleBar } from "../charts.js";
import { effectiveMapping, prefs, state } from "../state.js";
import { $, alertBox, esc, icon, num, pct, refreshIcons, verdictBadge } from "../ui.js";

let current = { upload: null, jobId: null, timer: null };

export function render(view) {
  const lim = state.meta.limits;
  view.innerHTML = `
    <div class="page-head"><div><h1>Batch Grading</h1>
      <p>Grade a whole class at once from a CSV file.</p></div>
      <button class="btn" id="b-sample">${icon("download")}Sample CSV</button></div>
    <section class="card card-pad" id="b-step1">
      <h2>1. Upload a CSV</h2>
      <p class="help" style="margin:4px 0 14px">Needs a question, a reference answer and a student answer column. Optional: maximum marks per row and a student ID.
        Limit: ${num(lim.batch_rows)} rows, ${lim.upload_mb} MB. Files are kept in server memory only while you work and are never written to disk.</p>
      <label class="dropzone" id="b-drop" for="b-file">
        ${icon("upload")}<div style="margin-top:6px"><strong>Choose a CSV file</strong> or drag it here</div>
        <input id="b-file" type="file" accept=".csv,text/csv" class="sr-only">
      </label>
      <div id="b-upload-msg" class="mt-sm"></div>
    </section>
    <section class="card card-pad mt" id="b-step2" hidden></section>
    <section class="card card-pad mt" id="b-step3" hidden></section>`;

  $("#b-sample", view).addEventListener("click", sampleCsv);
  const input = $("#b-file", view);
  const drop = $("#b-drop", view);
  input.addEventListener("change", () => input.files[0] && upload(view, input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(view, e.dataTransfer.files[0]));
  refreshIcons();
  return () => { clearTimeout(current.timer); current = { upload: null, jobId: null, timer: null }; };
}

function sampleCsv() {
  const rows = [["student_id", "question", "reference_answer", "student_answer", "max_marks"]];
  state.meta.examples.forEach((e, i) => rows.push([`S${String(i + 1).padStart(2, "0")}`, e.question, e.reference, e.answer, "2"]));
  rows.push(["S06", state.meta.examples[0].question, state.meta.examples[0].reference, "", "2"]);
  const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\r\n");
  saveBlob(new Blob([csv], { type: "text/csv" }), "gradesense_sample.csv");
}

async function upload(view, file) {
  const msg = $("#b-upload-msg", view);
  $("#b-step2", view).hidden = true;
  $("#b-step3", view).hidden = true;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    msg.innerHTML = alertBox("error", "Please choose a .csv file.");
    refreshIcons();
    return;
  }
  msg.innerHTML = `<div class="inline"><span class="spinner dark"></span>Reading ${esc(file.name)}…</div>`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    current.upload = await api("/api/batch/upload", { method: "POST", body: fd });
  } catch (e) {
    msg.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  msg.innerHTML = alertBox("success", `Loaded <strong>${esc(current.upload.filename)}</strong>: ${num(current.upload.rows)} rows, ${current.upload.columns.length} columns.`);
  renderMapping(view);
  refreshIcons();
}

function renderMapping(view) {
  const up = current.upload;
  const step = $("#b-step2", view);
  step.hidden = false;
  const opts = (sel, optional) => `${optional ? `<option value="">(not used)</option>` : `<option value="">Choose a column…</option>`}${
    up.columns.map((c) => `<option value="${esc(c)}" ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("")}`;
  const fields = [
    ["question", "Question", false], ["reference", "Reference answer", false], ["answer", "Student answer", false],
    ["max_marks", "Maximum marks (per row)", true], ["student_id", "Student ID (kept in output)", true],
  ];
  step.innerHTML = `<h2>2. Check the data and map the columns</h2>
    <div class="table-wrap mt-sm"><table><thead><tr>${up.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${up.preview.map((r) => `<tr>${up.columns.map((c) => `<td class="cell-text">${esc(String(r[c] ?? "").slice(0, 140))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="help mt-sm">Showing the first ${up.preview.length} of ${num(up.rows)} rows.</p>
    <div class="map-grid mt">${fields.map(([k, n, opt]) => `<div><label class="field" for="map-${k}">${n}${opt ? "" : " *"}</label>
      <select id="map-${k}">${opts(up.suggested_mapping[k], opt)}</select></div>`).join("")}</div>
    <div class="inline mt">
      <label for="b-max">Default maximum marks</label>
      <input type="number" id="b-max" min="0.5" max="100" step="0.5" value="${prefs().defaultMaxMarks}">
      <label class="check"><input type="checkbox" id="b-fb" checked>Concept feedback per answer (slower)</label>
    </div>
    <div id="b-map-err" class="field-error" role="alert"></div>
    <div class="btn-row mt"><button class="btn primary" id="b-start">${icon("play")}Grade ${num(up.rows)} answers</button></div>`;
  $("#b-start", step).addEventListener("click", () => start(view));
  refreshIcons();
}

async function start(view) {
  const get = (k) => $(`#map-${k}`, view).value;
  const mapping = { question: get("question"), reference: get("reference"), answer: get("answer"),
                    max_marks: get("max_marks"), student_id: get("student_id") };
  const err = $("#b-map-err", view);
  const missing = ["question", "reference", "answer"].filter((k) => !mapping[k]);
  if (missing.length) { err.textContent = "Choose a column for question, reference answer and student answer."; return; }
  if (new Set([mapping.question, mapping.reference, mapping.answer]).size < 3) {
    err.textContent = "Question, reference answer and student answer must be different columns."; return;
  }
  err.textContent = "";
  const btn = $("#b-start", view);
  btn.disabled = true;
  try {
    const res = await api("/api/batch/start", { method: "POST", body: {
      upload_id: current.upload.upload_id, mapping, default_max_marks: Number($("#b-max", view).value),
      with_feedback: $("#b-fb", view).checked, marks_mapping: effectiveMapping(),
    } });
    current.jobId = res.job_id;
    poll(view);
  } catch (e) {
    err.textContent = e.message;
    btn.disabled = false;
  }
}

async function poll(view) {
  const step = $("#b-step3", view);
  if (!step) return;
  step.hidden = false;
  let job;
  try {
    job = await api(`/api/batch/jobs/${current.jobId}`);
  } catch (e) {
    step.innerHTML = alertBox("error", esc(e.message));
    refreshIcons();
    return;
  }
  if (job.status === "running") {
    step.innerHTML = `<h2>3. Grading…</h2><div class="bar blue mt-sm"><div style="width:${(job.progress * 100).toFixed(1)}%"></div></div>
      <p class="help mt-sm">${num(job.done)} of ${num(job.total)} rows processed${job.invalid ? ` · ${num(job.invalid)} invalid rows skipped` : ""}.</p>`;
    current.timer = setTimeout(() => poll(view), 800);
    return;
  }
  if (job.status === "failed") {
    step.innerHTML = alertBox("error", esc(job.message));
    refreshIcons();
    return;
  }
  renderResults(step, job);
}

function renderResults(step, job) {
  const s = job.summary;
  const pageCols = job.columns.filter((c) => !["row_error"].includes(c));
  step.innerHTML = `<div class="card-head"><div><h2>3. Results</h2><p>${num(s.graded)} graded · ${num(job.invalid)} invalid · ${num(job.total)} rows in ${esc(job.filename)}</p></div>
      <button class="btn primary" id="b-dl">${icon("download")}Download graded CSV</button></div>
    <div class="grid grid-4">
      <div class="card stat"><div class="stat-body"><div class="stat-label">Average marks</div><div class="stat-value sm">${s.average_fraction === null ? "-" : pct(s.average_fraction)}</div><div class="stat-note">of maximum marks</div></div></div>
      <div class="card stat"><div class="stat-body"><div class="stat-label">Correct</div><div class="stat-value sm">${num(s.verdicts["Correct"])}</div><div class="stat-note">${num(s.verdicts["Partially Correct"])} partially correct</div></div></div>
      <div class="card stat"><div class="stat-body"><div class="stat-label">Incorrect</div><div class="stat-value sm">${num(s.verdicts["Incorrect"])}</div><div class="stat-note">contradictory, irrelevant or non-answer</div></div></div>
      <div class="card stat"><div class="stat-body"><div class="stat-label">Flagged for review</div><div class="stat-value sm">${num(s.needs_review)}</div><div class="stat-note">low model confidence</div></div></div>
    </div>
    <div class="grid grid-2 mt">
      <div><h3>Score distribution</h3><div class="chart-box short mt-sm"><canvas role="img" aria-label="Histogram of marks as a percentage of maximum marks"></canvas></div></div>
      <div><h3>Row problems</h3>${job.errors.length ? `<div class="table-wrap mt-sm" style="max-height:220px;overflow:auto"><table><thead><tr><th>CSV row</th><th>Problem (row not graded)</th></tr></thead><tbody>
        ${job.errors.map((e) => `<tr><td class="num">${e.row}</td><td>${esc(e.message)}</td></tr>`).join("")}</tbody></table></div>`
        : `<p class="help mt-sm">Every row was valid and graded.</p>`}</div>
    </div>
    <h3 class="mt">Graded answers</h3>
    <div class="table-wrap mt-sm" style="max-height:520px;overflow:auto"><table><thead><tr>${pageCols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>
      ${job.rows.map((r) => `<tr>${pageCols.map((c) => {
        const v = r[c];
        if (c === "verdict" && v) return `<td>${verdictBadge(v)}</td>`;
        if (c === "row_status") return `<td>${v === "graded" ? `<span class="badge green">graded</span>` : `<span class="badge red" title="${esc(r.row_error)}">${esc(v)}</span>`}</td>`;
        if (c === "model_confidence" && v !== "") return `<td class="num">${pct(Number(v))}</td>`;
        return `<td class="cell-text">${esc(String(v ?? "").slice(0, 200))}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table></div>
    ${job.rows.length < job.total ? `<p class="help mt-sm">Showing the first ${num(job.rows.length)} rows. The download contains all rows.</p>` : ""}`;
  $("#b-dl", step).addEventListener("click", async () => {
    try { await download(`/api/batch/jobs/${job.id}/download`, null, "graded_answers.csv"); }
    catch (e) { alert(e.message); }
  });
  simpleBar($("canvas", step), {
    labels: s.marks_distribution.map((b) => b.bin), data: s.marks_distribution.map((b) => b.count),
    yTitle: "Answers", xTitle: "Marks awarded (% of maximum)",
  });
  refreshIcons();
}

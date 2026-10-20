import { renderWorkspace } from "../components/workspace.js";
import { effectiveMapping, state } from "../state.js";
import { $, esc, icon, refreshIcons } from "../ui.js";

export function render(view) {
  const m = effectiveMapping() || {};
  const row = (k, n) => `<tr><td>${n}</td><td class="num">${Math.round((m[k] ?? 0) * 100)}%</td></tr>`;
  view.innerHTML = `
    <div class="page-head"><div><h1>Grade One</h1>
      <p>Grade a single answer and see why the model graded it that way.</p></div></div>
    <div id="workspace"></div>
    <div class="grid grid-2 mt">
      <section class="card card-pad">
        <h2>How a grade is produced</h2>
        <ol style="padding-left:18px;margin:10px 0 0;color:#334155">
          <li>The question, reference answer and student answer are joined as <code>[CLS] question + reference [SEP] student answer [SEP]</code>, exactly as in training.</li>
          <li>The fine-tuned DeBERTa-v3 cross-encoder predicts one of five labels: Correct, Partially Correct, Contradictory, Irrelevant or Non-Answer.</li>
          <li>The label maps to a verdict (Correct / Partially Correct / Incorrect) and to marks using the table on the right.</li>
          <li>Key concepts from the reference answer are checked against the student answer, and each word is tested by removing it and re-scoring (occlusion).</li>
        </ol>
      </section>
      <section class="card card-pad">
        <div class="card-head"><div><h2>Marks mapping</h2><p>Share of maximum marks for each model label.</p></div>
          <a class="btn sm" href="#/config">${icon("settings")}Change</a></div>
        <div class="table-wrap"><table><thead><tr><th>Model label</th><th class="num">Marks awarded</th></tr></thead><tbody>
          ${row("correct", "Correct")}${row("partially_correct_incomplete", "Partially Correct")}${row("contradictory", "Contradictory")}
          ${row("irrelevant", "Irrelevant")}${row("non_domain", "Non-Answer")}</tbody></table></div>
      </section>
    </div>`;
  renderWorkspace($("#workspace", view), { showOptions: true });
  refreshIcons();
  return () => { state.draft = null; };
}

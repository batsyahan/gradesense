import { state } from "../state.js";
import { esc } from "../ui.js";

const UPDATED = "20 October 2026";

export function privacy(view) {
  const owner = (state.meta.profile && state.meta.profile.name) || "the project author";
  view.innerHTML = `
    <div class="page-head"><div><h1>Privacy Policy</h1><p>Last updated ${UPDATED}</p></div></div>
    <section class="card card-pad prose">
      <p>GradeSense is a student academic project run by ${esc(owner)}. This page explains exactly what happens to the data you enter.</p>
      <h2>What GradeSense processes</h2>
      <ul>
        <li><strong>Text you submit</strong>: questions, reference answers and student answers, typed in or uploaded as CSV files.
        This text is sent to the GradeSense server and processed by the grading models.</li>
        <li><strong>A random browser ID</strong>: generated in your browser and stored in its local storage. It contains no personal
        information and is used only to show you your own grading history.</li>
        <li><strong>Preferences</strong> (marks mapping, display options) are stored only in your browser.</li>
      </ul>
      <h2>What is stored, and for how long</h2>
      <ul>
        <li>Each grading (the submitted text, the grade and a timestamp) is saved in a database on the server, linked to your browser ID,
        when grading history is enabled. Only the total number of gradings is shown to other visitors.</li>
        <li>You can export or delete your records at any time on the <a href="#/history">Grading History</a> page.</li>
        <li>On free hosting the database is wiped whenever the server restarts, so records should be treated as temporary.</li>
        <li>Uploaded CSV files are kept in server memory only while you work with them and are discarded within an hour. They are never
        written to disk.</li>
      </ul>
      <h2>What GradeSense does not do</h2>
      <ul>
        <li>No accounts, no sign-in and no collection of names, emails or contact details.</li>
        <li>No advertising, analytics or tracking scripts.</li>
        <li>Submitted text is not used to train models and is not shared or sold.</li>
      </ul>
      <h2>Third parties</h2>
      <ul>
        <li>The chart library and icons load from the jsDelivr CDN, which receives your IP address like any web request.</li>
        <li>The public version is hosted on Hugging Face Spaces, whose infrastructure processes requests under its own privacy policy.</li>
      </ul>
      <h2>Please avoid personal data</h2>
      <p>Do not enter names, roll numbers or other personal information about students. Use anonymised IDs in batch files.</p>
      <h2>Security</h2>
      <p>Records are separated by a random browser ID, which is not a password. GradeSense makes no claim of certified security or
      regulatory compliance and should not be used for sensitive or official assessment data.</p>
      <h2>Contact</h2>
      <p>Questions about this policy can be raised with ${esc(owner)} through the project's repository or the institution.</p>
    </section>`;
}

export function terms(view) {
  view.innerHTML = `
    <div class="page-head"><div><h1>Terms and Conditions</h1><p>Last updated ${UPDATED}</p></div></div>
    <section class="card card-pad prose">
      <h2 style="margin-top:0">Purpose</h2>
      <p>GradeSense is an academic demonstration of automatic short-answer grading. It is provided free of charge for learning,
      research and evaluation.</p>
      <h2>No guarantee of accuracy</h2>
      <p>Grades are predictions by a machine-learning model trained on school science answers. They can be wrong, especially for other
      subjects, long answers or unusual wording. Results are provided "as is", without warranty of any kind.</p>
      <h2>Not for high-stakes decisions</h2>
      <p>Do not use GradeSense as the sole basis for official marks, admissions, certification or any decision that significantly affects
      a person. A qualified teacher should review every grade that matters.</p>
      <h2>Your content</h2>
      <p>You are responsible for the text you submit and must have the right to use it. Do not submit personal data, confidential
      material or unlawful content. See the <a href="#/privacy">Privacy Policy</a> for how submitted text is handled.</p>
      <h2>Acceptable use</h2>
      <p>Do not attempt to overload, disrupt or gain unauthorised access to the service, or use it to process large volumes of data
      automatically. The service may be limited, changed or taken offline at any time.</p>
      <h2>Third-party components</h2>
      <p>GradeSense uses the SciEntsBank dataset (CC BY 4.0), the DeBERTa-v3 model (MIT licence) and other open-source software under their
      own licences, listed on the <a href="#/about">About</a> page.</p>
      <h2>Liability</h2>
      <p>To the extent permitted by law, the author is not liable for any loss arising from the use of GradeSense or reliance on its output.</p>
    </section>`;
}

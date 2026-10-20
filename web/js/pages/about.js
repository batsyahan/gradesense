import { state } from "../state.js";
import { esc } from "../ui.js";

export function render(view) {
  const p = state.meta.profile || {};
  view.innerHTML = `
    <div class="page-head"><div><h1>About GradeSense</h1>
      <p>A Deep Learning and NLP mini-project${p.name ? ` by ${esc(p.name)}` : ""}.</p></div></div>
    <section class="card card-pad prose">
      <h2 style="margin-top:0">Objective</h2>
      <p>Teachers spend hours marking short written answers, and keyword-based tools fail when a student says the right thing in
      different words. GradeSense grades short answers by comparing the <em>meaning</em> of a student answer with a reference
      answer, awards marks, and explains the grade with concept-level feedback.</p>

      <h2>Model architecture</h2>
      <p>The grader is <strong>DeBERTa-v3-base</strong> (184M parameters) fine-tuned as a <strong>cross-encoder</strong>. The question,
      reference answer and student answer are read together as one sequence
      (<code>[CLS] question + reference [SEP] student answer [SEP]</code>, up to 256 tokens), so self-attention can compare individual
      words across the two answers - which is what lets it notice contradictions such as "higher" versus "lower". A linear layer on the
      [CLS] representation predicts one of five labels. Training used AdamW (learning rate 2e-5, batch 16, linear warmup and decay),
      class-weighted cross-entropy, and kept the epoch with the best validation macro-F1.</p>
      <p>Feedback uses two further components: spaCy extracts key concepts from the reference answer, and Sentence-BERT
      (all-MiniLM-L6-v2) checks which of them appear in the student answer. Word highlights come from occlusion: each word is removed in
      turn and the answer is re-scored to see how much the expected marks change.</p>

      <h2>Dataset</h2>
      <p><strong>SciEntsBank</strong> from SemEval-2013 Task 7 (Student Response Analysis): about 10,800 answers by school students
      to 197 science questions across 15 domains, each labelled by experts.</p>

      <h2>Label scheme</h2>
      <div class="table-wrap"><table><thead><tr><th>Model label</th><th>Meaning</th><th>Verdict shown</th><th class="num">Default marks</th></tr></thead><tbody>
        <tr><td>Correct</td><td>Complete and correct paraphrase of the reference</td><td>Correct</td><td class="num">100%</td></tr>
        <tr><td>Partially Correct</td><td>Correct but incomplete</td><td>Partially Correct</td><td class="num">50%</td></tr>
        <tr><td>Contradictory</td><td>States something that contradicts the reference</td><td>Incorrect</td><td class="num">0%</td></tr>
        <tr><td>Irrelevant</td><td>On topic but does not answer the question</td><td>Incorrect</td><td class="num">0%</td></tr>
        <tr><td>Non-Answer</td><td>No attempt, e.g. "I don't know" or off-topic text</td><td>Incorrect</td><td class="num">0%</td></tr>
      </tbody></table></div>

      <h2>Baselines</h2>
      <ul>
        <li><strong>TF-IDF + logistic regression</strong>: word-overlap and TF-IDF features (two variants).</li>
        <li><strong>Siamese BiLSTM</strong>: deep learning trained from scratch on the training answers, without pretraining.</li>
        <li><strong>SBERT (no fine-tuning) + logistic regression</strong>: frozen pretrained sentence embeddings.</li>
        <li><strong>BERT (fine-tuned)</strong>: planned; shown as "Not evaluated" until it is trained.</li>
      </ul>

      <h2>Evaluation splits</h2>
      <p>Models are trained on 90% of the official training split; the other 10% (stratified, seed 42) is the validation set used to pick
      the best epoch. Three official test sets measure generalisation: <strong>Unseen Answers</strong> (new answers to known questions),
      <strong>Unseen Questions</strong> (new questions from known domains) and <strong>Unseen Domains</strong> (new science topics). The main
      metric is macro-F1 over the five classes; weighted-F1 and accuracy are also reported.</p>

      <h2>Known limitations</h2>
      <ul>
        <li>Trained only on short school science answers in English. Grades for other subjects or long answers are unreliable.</li>
        <li>Performance drops on questions and domains not seen in training.</li>
        <li>"Partially correct" is the hardest class, and Non-Answer has very few training examples.</li>
        <li>Confidence scores are uncalibrated softmax probabilities; fine-tuned models are often over-confident.</li>
        <li>Concept coverage and highlights are heuristics. Sentence embeddings place opposites (higher / lower) close together, so a
        concept can look "covered" in a contradictory answer.</li>
        <li>A fluent answer that copies the reference wording can receive more credit than it deserves. GradeSense should support
        a teacher, not replace one.</li>
      </ul>

      <h2>Citations and licences</h2>
      <ul>
        <li>Dzikovska et al. (2013). <em>SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment
        Challenge.</em> SciEntsBank dataset via <a href="https://huggingface.co/datasets/nkazi/SciEntsBank" target="_blank" rel="noopener">nkazi/SciEntsBank</a> (CC BY 4.0).</li>
        <li>He, Gao and Chen (2021). <em>DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled
        Embedding Sharing.</em> <code>microsoft/deberta-v3-base</code> (MIT licence).</li>
        <li>Reimers and Gurevych (2019). <em>Sentence-BERT.</em> <code>sentence-transformers/all-MiniLM-L6-v2</code> (Apache 2.0).</li>
        <li>spaCy (MIT), PyTorch (BSD), Hugging Face Transformers (Apache 2.0), FastAPI (MIT), Chart.js (MIT), Lucide icons (ISC).</li>
      </ul>
      <p class="mt">Version ${esc(state.meta.version)} · <a href="#/privacy">Privacy Policy</a> · <a href="#/terms">Terms and Conditions</a></p>
    </section>`;
}

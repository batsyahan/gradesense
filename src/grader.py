"""
GradeSense - grading engine.

- DeBERTa-v3 cross-encoder (fine-tuned on SciEntsBank, 5 labels) -> label, probabilities, marks
- Concept coverage (spaCy concepts + Sentence-BERT similarity)    -> covered / partially covered / missing
- Occlusion attribution on expected credit                         -> which answer words raised or lowered the score
- Rule-based feedback built from the three signals above

The model input format matches training exactly:
    [CLS] question + " " + reference [SEP] student answer [SEP], truncated to max_length tokens.
"""
import os
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "deberta_5way")

# 5-way label order used in training (src/data.py) - never reorder
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]

LABEL_NAMES = {
    "correct": "Correct",
    "contradictory": "Contradictory",
    "partially_correct_incomplete": "Partially Correct",
    "irrelevant": "Irrelevant",
    "non_domain": "Non-Answer",
}
# Student-facing 3-category verdict; the 5-way label is always kept alongside it
VERDICT3 = {
    "correct": "Correct",
    "partially_correct_incomplete": "Partially Correct",
    "contradictory": "Incorrect",
    "irrelevant": "Incorrect",
    "non_domain": "Incorrect",
}
DEFAULT_MARKS = {
    "correct": 1.0,
    "partially_correct_incomplete": 0.5,
    "contradictory": 0.0,
    "irrelevant": 0.0,
    "non_domain": 0.0,
}

# Kept for the older Streamlit app (app/app.py)
LABEL_DISPLAY = {
    "correct": "Correct",
    "contradictory": "Contradictory",
    "partially_correct_incomplete": "Partially correct",
    "irrelevant": "Irrelevant",
    "non_domain": "Non-domain",
}
MARK_FRACTION = DEFAULT_MARKS
FEEDBACK = {
    "correct": "Well done - the answer matches the meaning of the reference answer.",
    "partially_correct_incomplete": "On the right track, but the answer is incomplete.",
    "contradictory": "The answer contradicts the reference answer - it states the opposite of what is expected.",
    "irrelevant": "The answer does not address what the question is asking.",
    "non_domain": "No real attempt at the question (e.g. 'I don't know' or off-topic text).",
}
REVIEW_THRESHOLD = 0.60
CONCEPT_THRESHOLD = 0.60
PARTIAL_THRESHOLD = 0.45
HIGHLIGHT_THRESHOLD = 0.05   # change in expected credit (fraction of full marks) that counts as meaningful

STOPWORDS = set("""
a an the is are was were be been being am do does did to of in on at by for with from as and or but if
then so that this these those it its it's they them their there here what which who whom when where why
how will would can could should may might must shall i you he she we me my your his her our not no
because each other also very just only than too into onto over under about after before more most some
any all both such own same so up down out off again further once has have had having
""".split())


def _words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _short(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _quote_list(items, n=4):
    items = [f"“{i}”" for i in items[:n]]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def normalise_mapping(mapping):
    """Return a complete label -> fraction mapping (0..1), falling back to defaults."""
    out = dict(DEFAULT_MARKS)
    for k, v in (mapping or {}).items():
        if k in out:
            out[k] = min(1.0, max(0.0, float(v)))
    return out


class Grader:
    def __init__(self, model_dir=MODEL_DIR, sbert_name="sentence-transformers/all-MiniLM-L6-v2",
                 device="cpu", max_length=256, batch_size=16):
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().eval().to(device)
        self.n_parameters = sum(p.numel() for p in self.model.parameters())

        self.sbert = None
        self.sbert_error = None
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(sbert_name, device="cpu")
        except Exception as e:
            self.sbert_error = str(e)
            print("Sentence-BERT unavailable, concept matching uses exact words only:", e)

        self.nlp = None
        self.spacy_error = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.spacy_error = str(e)
            print("spaCy model unavailable, using simple keyword concepts:", e)

    # ================= core model =================
    @torch.no_grad()
    def _probs(self, firsts, seconds, batch_size=None):
        bs = batch_size or self.batch_size
        out = []
        for i in range(0, len(firsts), bs):
            enc = self.tokenizer(firsts[i:i + bs], seconds[i:i + bs], truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out.append(torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def probs(self, question, reference, answers):
        first = f"{question} {reference}".strip()
        return self._probs([first] * len(answers), [str(a) for a in answers])

    def probs_rows(self, questions, references, answers):
        firsts = [f"{q} {r}".strip() for q, r in zip(questions, references)]
        return self._probs(firsts, [str(a) for a in answers])

    # ================= concepts =================
    def _lemmas(self, text):
        if self.nlp is not None:
            return {t.lemma_.lower() for t in self.nlp(str(text)) if not t.is_punct}
        return set(_words(text))

    def _word_lemmas(self, words):
        """Lemma set for each whitespace-separated answer word."""
        out = []
        for w in words:
            clean = re.sub(r"[^\w'-]", "", w)
            out.append(self._lemmas(clean) if clean else set())
        return out

    def extract_concepts(self, reference, question="", max_concepts=8):
        """Key ideas in the reference answer that the question itself doesn't already give away."""
        q_words = set(_words(question))
        concepts = []
        if self.nlp is not None:
            doc = self.nlp(str(reference))
            for chunk in doc.noun_chunks:
                toks = [t.text for t in chunk if not t.is_stop and not t.is_punct]
                if toks:
                    concepts.append(" ".join(toks))
            for t in doc:
                if t.pos_ in ("VERB", "ADJ", "ADV", "NUM") and not t.is_stop and len(t.text) > 2:
                    concepts.append(t.text)
        else:
            concepts = [w for w in _words(reference) if w not in STOPWORDS and len(w) > 2]

        seen, unique = set(), []
        for c in concepts:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        new_info = [c for c in unique if not set(_words(c)) <= q_words]
        return (new_info or unique)[:max_concepts]

    def analyze_concepts(self, reference, answer, question="", label=None,
                         covered_threshold=CONCEPT_THRESHOLD, partial_threshold=PARTIAL_THRESHOLD):
        """Each concept gets status Covered / Partially Covered / Missing plus the answer word
        positions it matched (used for highlighting). This is a heuristic, not proof of understanding."""
        concepts = self.extract_concepts(reference, question)
        if not concepts:
            return []
        ans_words = str(answer).split()
        word_lemmas = self._word_lemmas(ans_words)
        ans_lemmas = set().union(*word_lemmas) if word_lemmas else set()

        spans, span_idx = [], []
        if self.sbert is not None and ans_words:
            for n in (1, 2, 3):
                for i in range(len(ans_words) - n + 1):
                    spans.append(" ".join(ans_words[i:i + n]))
                    span_idx.append(list(range(i, i + n)))
            span_emb = self.sbert.encode(spans, normalize_embeddings=True, convert_to_numpy=True)
            con_emb = self.sbert.encode(concepts, normalize_embeddings=True, convert_to_numpy=True)
            sims = con_emb @ span_emb.T

        results = []
        for k, c in enumerate(concepts):
            c_lemmas = self._lemmas(c) - STOPWORDS
            if c_lemmas and c_lemmas <= ans_lemmas:
                idx = [i for i, wl in enumerate(word_lemmas) if wl & c_lemmas]
                results.append({"concept": c, "status": "Covered", "score": 1.0, "match": "exact wording",
                                "method": "exact", "word_indices": idx, "note": ""})
                continue
            if spans:
                best = int(np.argmax(sims[k]))
                score = float(sims[k][best])
                if score >= covered_threshold:
                    status = "Covered"
                elif score >= partial_threshold:
                    status = "Partially Covered"
                else:
                    status = "Missing"
                note = ""
                if status == "Covered" and label == "contradictory":
                    # embeddings place opposites close together ("higher"/"lower"), so a similar phrase
                    # in a contradictory answer is mentioned, not necessarily stated correctly
                    status, note = "Partially Covered", "mentioned, but the answer may state it incorrectly"
                results.append({"concept": c, "status": status, "score": round(score, 3),
                                "match": spans[best] if status != "Missing" else "",
                                "method": "semantic", "word_indices": span_idx[best] if status != "Missing" else [],
                                "note": note})
            else:
                results.append({"concept": c, "status": "Missing", "score": 0.0, "match": "",
                                "method": "exact-only", "word_indices": [], "note": ""})
        return results

    # Kept for the Streamlit app
    def concept_coverage(self, reference, answer, question=""):
        return [{"concept": c["concept"], "covered": c["status"] == "Covered", "score": c["score"],
                 "match": c["match"]} for c in self.analyze_concepts(reference, answer, question)]

    # ================= attribution =================
    def credit_occlusion(self, question, reference, answer, mapping, max_words=40):
        """Expected credit = sum_k P(label_k) * marks_fraction_k.
        Remove one word at a time; delta = credit(full answer) - credit(answer without the word).
        delta > 0: the word raised the score; delta < 0: it lowered the score."""
        words = str(answer).split()
        if not words or len(words) > max_words:
            return None
        frac = np.array([mapping[l] for l in LABELS])
        base = float(self.probs(question, reference, [answer])[0] @ frac)
        variants = [" ".join(words[:k] + words[k + 1:]) for k in range(len(words))]
        credits = self.probs(question, reference, variants) @ frac
        return [(w, float(base - c)) for w, c in zip(words, credits)]

    # Kept for the Streamlit app
    def word_importance(self, question, reference, answer, label_index, max_words=60):
        words = str(answer).split()
        if len(words) < 2 or len(words) > max_words:
            return [(w, 0.0) for w in words]
        base = self.probs(question, reference, [answer])[0][label_index]
        variants = [" ".join(words[:k] + words[k + 1:]) for k in range(len(words))]
        p = self.probs(question, reference, variants)[:, label_index]
        return [(w, float(base - pk)) for w, pk in zip(words, p)]

    @staticmethod
    def build_highlights(answer, concepts, occlusion, threshold=HIGHLIGHT_THRESHOLD):
        words = str(answer).split()
        covered_idx, partial_idx = set(), set()
        for c in concepts:
            if c["status"] == "Covered":
                covered_idx.update(c["word_indices"])
            elif c["status"] == "Partially Covered":
                partial_idx.update(c["word_indices"])
        tokens = []
        for i, w in enumerate(words):
            delta = occlusion[i][1] if occlusion else 0.0
            if occlusion and delta <= -threshold:
                status = "incorrect"
            elif occlusion and delta >= threshold:
                status = "supported"
            elif i in partial_idx:
                status = "partial"
            elif i in covered_idx:
                status = "supported"
            else:
                status = "neutral"
            tokens.append({"text": w, "status": status, "delta": round(delta, 4)})
        if occlusion:
            method = ("Occlusion on expected credit (each word removed in turn and the answer re-scored by the "
                      "model) combined with concept matching. A heuristic view of what influenced the model, "
                      "not a definitive explanation of its reasoning.")
        else:
            method = ("Concept matching only (answer too long for word-by-word occlusion, or occlusion "
                      "switched off). Highlights show which words matched reference concepts.")
        return {"method": method, "occlusion": bool(occlusion), "tokens": tokens}

    # ================= feedback =================
    @staticmethod
    def build_feedback(question, reference, label, verdict, confidence, marks, max_marks, concepts, highlights):
        covered = [c["concept"] for c in concepts if c["status"] == "Covered"]
        partial = [c["concept"] for c in concepts if c["status"] == "Partially Covered"]
        missing = [c["concept"] for c in concepts if c["status"] == "Missing"]
        lowered = [t["text"].strip(".,;:!?") for t in (highlights or {}).get("tokens", []) if t["status"] == "incorrect"]

        got_right, not_there, wrong, improve = [], [], [], []

        if covered:
            got_right.append(f"Addresses the key idea{'s' if len(covered) > 1 else ''} {_quote_list(covered)}.")
        if partial and label != "contradictory":
            got_right.append(f"Touches on {_quote_list(partial)}, but not clearly or completely.")
        if label == "correct" and not covered:
            got_right.append("Expresses the meaning of the reference answer in different words.")

        if missing:
            not_there.append(f"Does not mention {_quote_list(missing)}.")
        if partial and label != "contradictory":
            not_there.append(f"{_quote_list(partial)} {'is' if len(partial) == 1 else 'are'} only partly explained.")

        if label == "contradictory":
            msg = "States the opposite of the reference answer."
            if lowered:
                msg += f" The words {_quote_list(lowered)} pushed the score down most."
            wrong.append(msg)
            if partial:
                wrong.append(f"Mentions {_quote_list(partial)}, but in a way that conflicts with the reference.")
        elif label == "irrelevant":
            wrong.append("Does not answer what the question asks.")
        elif label == "non_domain":
            wrong.append("Is not an attempt at the question (for example 'I don't know' or off-topic text).")
        elif lowered and label == "partially_correct_incomplete":
            wrong.append(f"The words {_quote_list(lowered)} lowered the model's score.")

        if label == "correct":
            improve.append("Nothing essential is missing for full marks."
                           if not missing else f"For a fuller answer, also mention {_quote_list(missing)}.")
        elif label == "partially_correct_incomplete":
            gaps = missing + partial
            improve.append(f"Add the missing idea{'s' if len(gaps) > 1 else ''}: {_quote_list(gaps)}."
                           if gaps else "Explain the idea in more detail and state the conclusion explicitly.")
            improve.append(f"Compare with the reference: “{_short(reference)}”")
        elif label == "contradictory":
            improve.append(f"Re-check the claim against the reference, which says: “{_short(reference)}”")
        elif label == "irrelevant":
            improve.append(f"Answer the question directly: “{_short(question)}”" if question.strip()
                           else "Answer the question that was asked directly.")
            if missing:
                improve.append(f"Build the answer around {_quote_list(missing)}.")
        else:
            improve.append(f"Attempt an answer using the key ideas {_quote_list(missing or covered or partial)}."
                           if (missing or covered or partial) else "Attempt an answer to the question.")

        summary = (f"{verdict}: the model's 5-class prediction is “{LABEL_NAMES[label]}” "
                   f"(confidence {confidence:.0%}). Awarded {marks:g} of {max_marks:g} marks.")
        return {"summary": summary, "got_right": got_right, "missing": not_there,
                "incorrect": wrong, "improve": improve}

    # ================= full analysis =================
    def analyze(self, question, reference, answer, max_marks=2, marks_mapping=None, explain=True,
                review_threshold=REVIEW_THRESHOLD, covered_threshold=CONCEPT_THRESHOLD,
                partial_threshold=PARTIAL_THRESHOLD, occlusion_max_words=40):
        mapping = normalise_mapping(marks_mapping)
        p = self.probs(question, reference, [answer])[0]
        i = int(np.argmax(p))
        label = LABELS[i]
        confidence = float(p[i])
        verdict = VERDICT3[label]
        marks = round(mapping[label] * float(max_marks), 2)
        expected_credit = float(sum(p[k] * mapping[l] for k, l in enumerate(LABELS)))

        concepts = self.analyze_concepts(reference, answer, question, label, covered_threshold, partial_threshold)
        occlusion = self.credit_occlusion(question, reference, answer, mapping, occlusion_max_words) if explain else None
        highlights = self.build_highlights(answer, concepts, occlusion)
        feedback = self.build_feedback(question, reference, label, verdict, confidence, marks,
                                       float(max_marks), concepts, highlights)
        return {
            "label": label,
            "label_name": LABEL_NAMES[label],
            "verdict": verdict,
            "probabilities": [{"label": l, "name": LABEL_NAMES[l], "p": round(float(p[k]), 4)}
                              for k, l in enumerate(LABELS)],
            "confidence": round(confidence, 4),
            "expected_credit": round(expected_credit, 4),
            "marks": marks,
            "max_marks": float(max_marks),
            "marks_mapping": mapping,
            "needs_review": confidence < review_threshold,
            "concepts": [{k: v for k, v in c.items() if k != "word_indices"} for c in concepts],
            "highlights": highlights,
            "feedback": feedback,
        }

    def grade_rows(self, questions, references, answers, max_marks, marks_mapping=None,
                   review_threshold=REVIEW_THRESHOLD, with_concepts=True):
        """Fast path for batch grading: one model pass per row plus (optionally) concept coverage."""
        mapping = normalise_mapping(marks_mapping)
        p = self.probs_rows(questions, references, answers)
        out = []
        for row, probs in enumerate(p):
            i = int(np.argmax(probs))
            label = LABELS[i]
            missing = []
            if with_concepts and label != "correct":
                missing = [c["concept"] for c in self.analyze_concepts(references[row], answers[row],
                                                                       questions[row], label)
                           if c["status"] != "Covered"]
            fb = FEEDBACK[label]
            if missing:
                fb += f" Not found in the answer: {', '.join(missing[:4])}."
            out.append({
                "label": label,
                "label_name": LABEL_NAMES[label],
                "verdict": VERDICT3[label],
                "confidence": round(float(probs[i]), 4),
                "marks": round(mapping[label] * float(max_marks[row]), 2),
                "max_marks": float(max_marks[row]),
                "needs_review": bool(probs[i] < review_threshold),
                "feedback": fb,
            })
        return out

    # Kept for the Streamlit app
    def grade(self, question, reference, answer, max_marks=2):
        p = self.probs(question, reference, [answer])[0]
        i = int(np.argmax(p))
        label = LABELS[i]
        return {
            "label": label,
            "label_index": i,
            "display": LABEL_DISPLAY[label],
            "confidence": float(p[i]),
            "probs": {LABEL_DISPLAY[l]: float(p[k]) for k, l in enumerate(LABELS)},
            "marks": MARK_FRACTION[label] * max_marks,
            "max_marks": max_marks,
            "needs_review": bool(p[i] < REVIEW_THRESHOLD),
            "feedback": FEEDBACK[label],
        }

    def grade_many(self, questions, references, answers, max_marks=2, batch_size=16, progress=None):
        firsts = [f"{q} {r}".strip() for q, r in zip(questions, references)]
        answers = [str(a) if a == a else "" for a in answers]
        n = len(answers)
        all_p = []
        for i in range(0, n, batch_size):
            all_p.append(self._probs(firsts[i:i + batch_size], answers[i:i + batch_size], batch_size))
            if progress:
                progress(min(1.0, (i + batch_size) / n))
        p = np.concatenate(all_p)
        idx = p.argmax(axis=1)
        labels = [LABELS[k] for k in idx]
        return {
            "verdict": [LABEL_DISPLAY[l] for l in labels],
            "confidence": p.max(axis=1).round(3),
            "marks": [MARK_FRACTION[l] * max_marks for l in labels],
            "needs_review": p.max(axis=1) < REVIEW_THRESHOLD,
        }

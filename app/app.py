"""
GradeSense - Streamlit demo.
Run from the project folder:  streamlit run app/app.py
"""
import html
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.grader import LABEL_DISPLAY, MODEL_DIR, Grader

RESULTS_DIR = os.path.join(ROOT, "results")

st.set_page_config(page_title="GradeSense", page_icon="📝", layout="wide")

COLORS = {
    "correct": "#2e7d32",
    "partially_correct_incomplete": "#ef8f00",
    "contradictory": "#c62828",
    "irrelevant": "#6d4c41",
    "non_domain": "#616161",
}

EXAMPLES = {
    "Pitch - paraphrased correct answer": (
        "What happens to the pitch when the string is tightened?",
        "When the string is tighter, the pitch will be higher.",
        "The pitch goes up when you tighten it."),
    "Pitch - contradiction": (
        "What happens to the pitch when the string is tightened?",
        "When the string is tighter, the pitch will be higher.",
        "The pitch will be lower."),
    "Salt and water - correct, different words": (
        "You used several methods to separate and identify the substances in mock rocks. "
        "How did you separate the salt from the water?",
        "The water was evaporated, leaving the salt.",
        "We left it in a dish until the water dried up and only salt was left."),
    "Salt and water - incomplete": (
        "You used several methods to separate and identify the substances in mock rocks. "
        "How did you separate the salt from the water?",
        "The water was evaporated, leaving the salt.",
        "Evaporate."),
    "Circuit - contradiction": (
        "If the middle bulb in a series circuit burns out, what happens to the other bulbs?",
        "If the middle bulb burns out, the other 2 bulbs will go out because the circuit is broken.",
        "The other bulbs will stay lit."),
    "Magnets - irrelevant": (
        "Why does a magnet stick to a car door?",
        "A car is made of a lot of iron or steel.",
        "Because magnets are fun to play with."),
}


# ---------------- model ----------------
if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    st.error(f"Model not found in {MODEL_DIR}. Copy the fine-tuned DeBERTa files there first.")
    st.stop()


@st.cache_resource(show_spinner="Loading the GradeSense model (first time only)...")
def get_grader():
    return Grader()


grader = get_grader()


# ---------------- helpers ----------------
def verdict_box(label, display, confidence):
    color = COLORS[label]
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{color}1a;"
        f"border-left:6px solid {color};margin-bottom:8px'>"
        f"<span style='font-size:1.5rem;font-weight:700;color:{color}'>{display}</span>"
        f"<span style='margin-left:12px;opacity:0.8'>confidence {confidence:.0%}</span></div>",
        unsafe_allow_html=True,
    )


def highlight_html(importances):
    if not importances:
        return ""
    peak = max(abs(v) for _, v in importances) or 1.0
    parts = []
    for word, v in importances:
        alpha = min(1.0, abs(v) / peak) * 0.55
        rgb = "46,125,50" if v > 0 else "198,40,40"
        parts.append(f"<span style='background:rgba({rgb},{alpha:.2f});padding:2px 4px;"
                     f"border-radius:4px;margin:1px;display:inline-block'>{html.escape(word)}</span>")
    return "<div style='font-size:1.1rem;line-height:2'>" + " ".join(parts) + "</div>"


def concept_chips(coverage):
    chips = []
    for c in coverage:
        if c["covered"]:
            style, icon = "background:#2e7d321f;border:1px solid #2e7d32;color:#2e7d32", "✓"
        else:
            style, icon = "background:#9e9e9e1f;border:1px dashed #9e9e9e;color:#757575", "✗"
        tip = f"similarity {c['score']:.2f}" + (f" - matched '{c['match']}'" if c["match"] else "")
        chips.append(f"<span title=\"{html.escape(tip)}\" style='{style};padding:4px 10px;border-radius:14px;"
                     f"margin:3px;display:inline-block'>{icon} {html.escape(c['concept'])}</span>")
    return "<div>" + " ".join(chips) + "</div>"


# ---------------- sidebar ----------------
st.sidebar.title("📝 GradeSense")
page = st.sidebar.radio("Go to", ["Grade One", "Batch Grade", "Model Insights"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** DeBERTa-v3-base, fine-tuned as a cross-encoder on SciEntsBank (SemEval-2013 Task 7)\n\n"
    "**Labels:** correct · partially correct · contradictory · irrelevant · non-domain\n\n"
    "**Marks:** correct = full, partially correct = half, otherwise 0"
)


# ================= PAGE 1: GRADE ONE =================
if page == "Grade One":
    st.title("Grade a single answer")
    st.caption("Transformer-based short answer grading with concept-level feedback")

    choice = st.selectbox("Load an example (or type your own below)", ["(blank)"] + list(EXAMPLES))
    dq, dr, da = EXAMPLES.get(choice, ("", "", ""))

    question = st.text_area("Question", value=dq, height=80, key=f"q_{choice}")
    reference = st.text_area("Reference answer (from the teacher)", value=dr, height=80, key=f"r_{choice}")
    answer = st.text_area("Student answer", value=da, height=80, key=f"a_{choice}")
    c1, c2 = st.columns(2)
    max_marks = c1.number_input("Maximum marks", min_value=1, max_value=10, value=2)
    explain = c2.checkbox("Show word-level explanation", value=True)

    if st.button("Grade answer", type="primary"):
        if not reference.strip() or not answer.strip():
            st.warning("Please enter at least a reference answer and a student answer.")
            st.stop()

        with st.spinner("Grading..."):
            r = grader.grade(question, reference, answer, max_marks)
            coverage = grader.concept_coverage(reference, answer, question)
            importances = grader.word_importance(question, reference, answer, r["label_index"]) if explain else []

        left, right = st.columns([3, 2])
        with left:
            verdict_box(r["label"], r["display"], r["confidence"])
            m1, m2 = st.columns(2)
            m1.metric("Marks", f"{r['marks']:g} / {r['max_marks']}")
            m2.metric("Confidence", f"{r['confidence']:.0%}")
            if r["needs_review"]:
                st.warning("Low confidence - recommend the teacher reviews this answer.")
            st.markdown(f"**Feedback:** {r['feedback']}")
            missing = [c["concept"] for c in coverage if not c["covered"]]
            if missing and r["label"] != "correct":
                st.markdown("**Key ideas not found in the answer:** " + ", ".join(f"*{m}*" for m in missing))
        with right:
            st.markdown("**Probability for each label**")
            st.bar_chart(pd.Series(r["probs"], name="probability"), horizontal=True, height=220)

        st.markdown("#### Key concepts from the reference answer")
        if coverage:
            st.markdown(concept_chips(coverage), unsafe_allow_html=True)
            st.caption("✓ = found in the student answer (exact word or similar meaning, via Sentence-BERT). "
                       "Hover a concept to see the similarity score.")
        else:
            st.caption("No key concepts could be extracted from the reference answer.")

        if explain and importances:
            st.markdown("#### Which words drove the verdict?")
            st.markdown(highlight_html(importances), unsafe_allow_html=True)
            st.caption(f"Green = supports the verdict '{r['display']}', red = argues against it. "
                       "Computed by removing each word and measuring the change in confidence (occlusion).")


# ================= PAGE 2: BATCH GRADE =================
elif page == "Batch Grade":
    st.title("Batch grading")
    st.markdown("Upload a CSV with the columns **question**, **reference_answer**, **student_answer** "
                "(and optionally **student_id**).")

    sample = pd.DataFrame(
        [(f"S{i + 1:02d}", q, r, a) for i, (q, r, a) in enumerate(EXAMPLES.values())],
        columns=["student_id", "question", "reference_answer", "student_answer"])
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_answers.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    max_marks = st.number_input("Maximum marks per answer", min_value=1, max_value=10, value=2)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        needed = {"question", "reference_answer", "student_answer"}
        if not needed <= set(df.columns):
            st.error(f"The CSV is missing these columns: {', '.join(sorted(needed - set(df.columns)))}")
            st.stop()
        st.write(f"{len(df)} answers loaded.")
        st.dataframe(df.head(10))

        if st.button("Grade all", type="primary"):
            bar = st.progress(0.0, text="Grading...")
            out = grader.grade_many(df["question"].fillna("").astype(str).tolist(),
                                    df["reference_answer"].fillna("").astype(str).tolist(),
                                    df["student_answer"].tolist(), max_marks,
                                    progress=lambda f: bar.progress(f, text=f"Grading... {f:.0%}"))
            bar.empty()
            graded = df.copy()
            for k, v in out.items():
                graded[k] = v
            st.session_state["graded"] = graded
            st.session_state["graded_max"] = max_marks

    if "graded" in st.session_state:
        graded, mm = st.session_state["graded"], st.session_state["graded_max"]
        st.markdown("### Results")
        a, b, c, d = st.columns(4)
        a.metric("Answers", len(graded))
        b.metric("Average marks", f"{graded['marks'].mean():.2f} / {mm}")
        c.metric("Fully correct", f"{(graded['verdict'] == 'Correct').mean():.0%}")
        d.metric("Flagged for review", int(graded["needs_review"].sum()))
        st.dataframe(graded)

        left, right = st.columns(2)
        with left:
            st.markdown("**Marks distribution**")
            st.bar_chart(graded["marks"].value_counts().sort_index())
        with right:
            st.markdown("**Verdicts**")
            st.bar_chart(graded["verdict"].value_counts())
        st.download_button("Download graded CSV", graded.to_csv(index=False), "graded_answers.csv",
                           "text/csv", type="primary")


# ================= PAGE 3: MODEL INSIGHTS =================
else:
    st.title("Model insights")
    path = os.path.join(RESULTS_DIR, "results.csv")
    if not os.path.exists(path):
        st.info("results/results.csv not found - run the training scripts first.")
        st.stop()

    res = pd.read_csv(path)
    schemes = sorted(res["scheme"].unique())
    scheme = st.selectbox("Label scheme", schemes, index=schemes.index("5way") if "5way" in schemes else 0)
    metric = st.radio("Metric", ["weighted_f1", "macro_f1", "accuracy"], horizontal=True)
    sub = res[res["scheme"] == scheme]
    order = [s for s in ["val", "test_ua", "test_uq", "test_ud"] if s in sub["split"].unique()]
    table = sub.pivot(index="model", columns="split", values=metric)[order]
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#2e7d3240"))
    st.caption("test_ua = unseen answers · test_uq = unseen questions · test_ud = unseen domains")
    st.bar_chart(table.drop(columns=["val"], errors="ignore").T, stack=False)

    hist_path = os.path.join(RESULTS_DIR, f"history_deberta_{scheme}.csv")
    if os.path.exists(hist_path):
        st.markdown("### DeBERTa training curve")
        hist = pd.read_csv(hist_path).set_index("epoch")
        h1, h2 = st.columns(2)
        h1.line_chart(hist["train_loss"])
        h2.line_chart(hist["val_macro_f1"])
        st.caption("Left: training loss. Right: validation macro-F1. Best epoch is saved (early stopping).")

    st.markdown("### Confusion matrices")
    models = sorted(sub["model"].unique())
    model_name = st.selectbox("Model", models, index=models.index("DeBERTa-v3") if "DeBERTa-v3" in models else 0)
    cols = st.columns(3)
    for col, split in zip(cols, ["test_ua", "test_uq", "test_ud"]):
        img = os.path.join(RESULTS_DIR, f"cm_{model_name}_{scheme}_{split}.png")
        if os.path.exists(img):
            col.image(img, caption=split)
        else:
            col.caption(f"No image for {split}")

    st.markdown("### Why the transformer wins: the pitch example")
    st.markdown(
        "| Student answer | TF-IDF | BiLSTM | DeBERTa |\n|---|---|---|---|\n"
        "| *The pitch will be lower.* (contradiction) | ❌ correct | ✅ contradictory | ✅ contradictory |\n"
        "| *The pitch goes up when you tighten it.* (paraphrase) | - | ❌ contradictory | ✅ correct |"
    )

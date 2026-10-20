"""
GradeSense dataset access for the Dataset Explorer and Error Analysis.
Loads SciEntsBank with exactly the same splits as training (src/data.py: 10% stratified validation, seed 42).
"""
import logging
import threading

import numpy as np

log = logging.getLogger("gradesense")

SPLIT_INFO = {
    "train": ("Training", "train", "Used to fit model weights: 90% of the official training split."),
    "val": ("Validation", "validation", "10% of the official training split (stratified, seed 42); "
                                        "used only to pick the best epoch."),
    "test_ua": ("Unseen Answers", "test", "New answers to questions that appear in training."),
    "test_uq": ("Unseen Questions", "test", "Questions never seen in training, same science domains."),
    "test_ud": ("Unseen Domains", "test", "Questions from science domains never seen in training."),
}


class DatasetStore:
    def __init__(self, scheme="5way"):
        self.scheme = scheme
        self.splits = None
        self.labels = None
        self.status = "not_loaded"
        self.error = None
        self._lock = threading.Lock()

    def start_background_load(self):
        threading.Thread(target=self.load, daemon=True, name="dataset-loader").start()

    def load(self):
        with self._lock:
            if self.splits is not None:
                return self.splits
            self.status = "loading"
            try:
                from src.data import label_names, load_splits
                self.splits = load_splits(scheme=self.scheme)
                self.labels = label_names(self.scheme)
                self.status = "loaded"
            except Exception as e:
                self.status, self.error = "error", str(e)
                log.exception("Dataset failed to load")
            return self.splits

    def require(self):
        if self.splits is None:
            self.load()
        if self.splits is None:
            raise RuntimeError(f"The dataset could not be loaded: {self.error}")
        return self.splits

    def overview(self, label_names):
        splits = self.require()
        out = []
        for key, (name, role, desc) in SPLIT_INFO.items():
            df = splits[key]
            counts = df["label_name"].value_counts()
            dist = [{"label": l, "name": label_names[i], "count": int(counts.get(l, 0))}
                    for i, l in enumerate(self.labels)]
            nonzero = [d["count"] for d in dist if d["count"] > 0]
            missing = {c: int((df[c].isna() | (df[c].astype(str).str.strip() == "")).sum())
                       for c in ("question", "reference_answer", "student_answer")}
            words = df["student_answer"].astype(str).str.split().str.len()
            out.append({
                "key": key, "name": name, "role": role, "description": desc,
                "rows": int(len(df)), "questions": int(df["question"].nunique()),
                "distribution": dist,
                "imbalance_ratio": round(max(nonzero) / min(nonzero), 1) if nonzero else None,
                "absent_labels": [d["name"] for d in dist if d["count"] == 0],
                "missing": missing,
                "answer_words_mean": round(float(words.mean()), 1),
                "answer_words_p95": int(np.percentile(words, 95)),
            })
        return out

    def samples(self, split, label=None, query="", offset=0, limit=20):
        splits = self.require()
        if split not in splits:
            raise KeyError(split)
        df = splits[split]
        if label:
            df = df[df["label_name"] == label]
        if query:
            q = query.lower()
            mask = (df["question"].str.lower().str.contains(q, regex=False)
                    | df["reference_answer"].str.lower().str.contains(q, regex=False)
                    | df["student_answer"].str.lower().str.contains(q, regex=False))
            df = df[mask]
        page = df.iloc[offset:offset + limit]
        return {"total": int(len(df)), "items": [
            {"id": r["id"], "question": r["question"], "reference_answer": r["reference_answer"],
             "student_answer": r["student_answer"], "label": r["label_name"]} for _, r in page.iterrows()]}

    def search_questions(self, query, limit=6):
        if self.splits is None or not query:
            return []
        q = query.lower()
        seen, out = set(), []
        for key in ("train", "test_ua", "test_uq", "test_ud"):
            df = self.splits[key]
            hits = df[df["question"].str.lower().str.contains(q, regex=False)]
            for question in hits["question"].unique():
                if question not in seen:
                    seen.add(question)
                    out.append({"question": question, "split": key, "split_name": SPLIT_INFO[key][0]})
                if len(out) >= limit:
                    return out
        return out

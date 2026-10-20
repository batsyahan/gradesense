"""
GradeSense analytics: reads the saved evaluation artifacts in results/ - nothing here is invented.

  results.csv                         -> accuracy, macro-F1, weighted-F1 per model / scheme / split
  preds_<model>_<scheme>_<split>.csv  -> confusion matrices, precision/recall, error analysis
  history_deberta_<scheme>.csv        -> training curve
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import resolve

TEST_SPLITS = ["test_ua", "test_uq", "test_ud"]
ALL_SPLITS = ["val"] + TEST_SPLITS
SPLIT_NAMES = {"train": "Training", "val": "Validation", "test_ua": "Unseen Answers",
               "test_uq": "Unseen Questions", "test_ud": "Unseen Domains"}

MODEL_INFO = {
    "TFIDF-Sim": {"name": "TF-IDF similarity + LR", "short": "TF-IDF Sim", "family": "Classical ML"},
    "TFIDF-Full": {"name": "TF-IDF + LR", "short": "TF-IDF + LR", "family": "Classical ML"},
    "BiLSTM": {"name": "Siamese BiLSTM", "short": "Siamese BiLSTM", "family": "Deep learning, no pretraining"},
    "SBERT-LR": {"name": "SBERT (No FT) + LR", "short": "SBERT (No FT)", "family": "Frozen pretrained embeddings"},
    "BERT": {"name": "BERT (FT)", "short": "BERT (FT)", "family": "Fine-tuned transformer"},
    "DeBERTa-v3": {"name": "DeBERTa-v3 (FT)", "short": "DeBERTa-v3 (FT)", "family": "Fine-tuned transformer"},
}
MODEL_ORDER = ["TFIDF-Sim", "TFIDF-Full", "BiLSTM", "SBERT-LR", "BERT", "DeBERTa-v3"]
CHART_MODELS = ["TFIDF-Full", "BiLSTM", "SBERT-LR", "BERT", "DeBERTa-v3"]


def _dir(cfg):
    return resolve(cfg["results_dir"])


def model_meta(key):
    info = MODEL_INFO.get(key, {"name": key, "short": key, "family": "Other"})
    return {"key": key, **info}


def read_results(cfg):
    path = _dir(cfg) / "results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["model", "scheme", "split", "accuracy", "macro_f1", "weighted_f1"])
    return pd.read_csv(path)


def known_models(df, scheme):
    present = list(dict.fromkeys(df.loc[df["scheme"] == scheme, "model"]))
    return [m for m in MODEL_ORDER if m in present or m in CHART_MODELS] + [m for m in present if m not in MODEL_ORDER]


def comparison(cfg, scheme, metric, chart_only=False):
    df = read_results(cfg)
    sub = df[df["scheme"] == scheme]
    keys = CHART_MODELS if chart_only else known_models(df, scheme)
    models = []
    for key in keys:
        rows = sub[sub["model"] == key]
        values = {}
        for split in ALL_SPLITS:
            r = rows[rows["split"] == split]
            values[split] = round(float(r.iloc[0][metric]), 4) if len(r) and metric in r else None
        models.append({**model_meta(key), "evaluated": any(values[s] is not None for s in TEST_SPLITS),
                       "values": values})
    return {"scheme": scheme, "metric": metric, "splits": [{"key": s, "name": SPLIT_NAMES[s]} for s in ALL_SPLITS],
            "models": models}


def preds_path(cfg, model, scheme, split):
    d = _dir(cfg)
    for name in (f"preds_{model}_{scheme}_{split}.csv", f"preds_{model.split('-')[0].lower()}_{scheme}_{split}.csv"):
        if (d / name).exists():
            return d / name
    return None


def load_preds(cfg, model, scheme, split):
    p = preds_path(cfg, model, scheme, split)
    if p is None:
        return None
    df = pd.read_csv(p)
    return df if {"label", "pred"} <= set(df.columns) else None


def confusion(cfg, model, scheme, split, labels):
    df = load_preds(cfg, model, scheme, split)
    if df is None:
        return None
    n = len(labels)
    cm = confusion_matrix(df["label"], df["pred"], labels=list(range(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    return {"model": model_meta(model), "split": split, "split_name": SPLIT_NAMES.get(split, split),
            "labels": labels, "counts": cm.tolist(), "normalized": np.round(norm, 4).tolist(),
            "row_totals": cm.sum(axis=1).tolist(), "total": int(cm.sum())}


def split_metrics(cfg, model, scheme, n_labels):
    """Per split: accuracy, macro-F1, weighted-F1 and macro-averaged precision/recall.
    Macro averages run over all classes; a class absent from a split contributes 0 (same rule as macro-F1)."""
    res = read_results(cfg)
    res = res[(res["scheme"] == scheme) & (res["model"] == model)]
    labels = list(range(n_labels))
    rows = []
    for split in ALL_SPLITS:
        row = {"split": split, "split_name": SPLIT_NAMES[split], "macro_f1": None, "accuracy": None,
               "weighted_f1": None, "precision": None, "recall": None, "support": None, "source": None}
        preds = load_preds(cfg, model, scheme, split)
        if preds is not None:
            y, p = preds["label"], preds["pred"]
            row.update({
                "macro_f1": round(f1_score(y, p, labels=labels, average="macro", zero_division=0), 4),
                "weighted_f1": round(f1_score(y, p, labels=labels, average="weighted", zero_division=0), 4),
                "accuracy": round(accuracy_score(y, p), 4),
                "precision": round(precision_score(y, p, labels=labels, average="macro", zero_division=0), 4),
                "recall": round(recall_score(y, p, labels=labels, average="macro", zero_division=0), 4),
                "support": int(len(preds)), "source": "predictions",
            })
        else:
            r = res[res["split"] == split]
            if len(r):
                row.update({"macro_f1": round(float(r.iloc[0]["macro_f1"]), 4),
                            "accuracy": round(float(r.iloc[0]["accuracy"]), 4),
                            "weighted_f1": round(float(r.iloc[0]["weighted_f1"]), 4),
                            "source": "results.csv"})
        rows.append(row)
    return {"model": model_meta(model), "scheme": scheme, "rows": rows,
            "averaging": "Precision and recall are macro-averaged over all 5 classes."}


def best_model(cfg, scheme):
    df = read_results(cfg)
    sub = df[(df["scheme"] == scheme) & (df["split"].isin(TEST_SPLITS))]
    if sub.empty:
        return None
    means = sub.groupby("model")["macro_f1"].mean().sort_values(ascending=False)
    key = means.index[0]
    rows = sub[sub["model"] == key]
    values = {r["split"]: round(float(r["macro_f1"]), 4) for _, r in rows.iterrows()}
    return {**model_meta(key), "mean_macro_f1": round(float(means.iloc[0]), 4), "values": values,
            "splits": [s for s in TEST_SPLITS if s in values]}


def training_history(cfg, scheme):
    path = _dir(cfg) / f"history_deberta_{scheme}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    best = int(df.loc[df["val_macro_f1"].idxmax(), "epoch"]) if len(df) else None
    return {"epochs": df["epoch"].tolist(), "train_loss": [round(x, 4) for x in df["train_loss"]],
            "val_macro_f1": [round(x, 4) for x in df["val_macro_f1"]], "best_epoch": best}


def artifact_status(cfg, model_dir):
    d = _dir(cfg)
    scheme = cfg["scheme"]

    def item(name, path, purpose):
        exists = path.exists()
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if exists and path.is_dir() else (
            path.stat().st_size if exists else None)
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exists else None
        return {"name": name, "path": os.path.relpath(path, resolve(".")), "exists": exists, "size": size,
                "modified": modified, "purpose": purpose}

    items = [
        item("Fine-tuned DeBERTa-v3", model_dir, "Grading model"),
        item("Results table", d / "results.csv", "Model comparison, summary cards"),
        item("DeBERTa training history", d / f"history_deberta_{scheme}.csv", "Training curve"),
    ]
    models = resolve("models")
    for key, fname in (("TFIDF-Full", f"tfidf_full_{scheme}.joblib"), ("BiLSTM", f"bilstm_{scheme}.pt"),
                       ("SBERT-LR", f"sbert_lr_{scheme}.joblib")):
        items.append(item(f"{MODEL_INFO[key]['name']} model", models / fname, "Model Playground"))
    df = read_results(cfg)
    for key in known_models(df, scheme):
        found = [s for s in ALL_SPLITS if preds_path(cfg, key, scheme, s) is not None]
        items.append({"name": f"{MODEL_INFO.get(key, {'name': key})['name']} predictions",
                      "path": f"{cfg['results_dir']}/preds_{key}_{scheme}_*.csv", "exists": bool(found),
                      "size": None, "modified": None,
                      "purpose": f"Confusion matrix, precision/recall, error analysis ({len(found)}/4 splits)"})
    return items


# ------------------------------------------------ error analysis
def error_category(t, p):
    if t == p:
        return "Correctly classified"
    if t == "correct":
        return "Under-credited: correct answer marked down"
    if p == "correct":
        return "Over-credited: wrong answer marked correct"
    if t == "contradictory":
        return "Missed contradiction"
    if p == "contradictory":
        return "False contradiction"
    if "partially_correct_incomplete" in (t, p):
        return "Partial-credit confusion"
    return "Irrelevant / non-answer confusion"


def error_frame(cfg, model, scheme, split, label_keys, split_frames):
    """Join saved predictions with the dataset text. Returns (DataFrame, message)."""
    preds = load_preds(cfg, model, scheme, split)
    if preds is None:
        return None, (f"No saved predictions for {model_meta(model)['name']} on {SPLIT_NAMES.get(split, split)}. "
                      "Run python scripts/build_baselines.py (baselines) or the Colab evaluation cell (DeBERTa).")
    source = split_frames.get(split)
    if source is None:
        return None, "The dataset is not loaded yet."
    if "id" not in preds.columns:
        return None, "This predictions file has no id column, so answers cannot be matched to the dataset."
    text = source[["id", "question", "reference_answer", "student_answer"]].drop_duplicates("id")
    df = preds.merge(text, on="id", how="left")
    df["true_label"] = [label_keys[int(i)] for i in df["label"]]
    df["pred_label"] = [label_keys[int(i)] for i in df["pred"]]
    df["category"] = [error_category(t, p) for t, p in zip(df["true_label"], df["pred_label"])]
    return df, None


def error_summary(df, label_keys, label_names):
    errors = df[df["true_label"] != df["pred_label"]]
    pairs = (errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False).head(10))
    per_class = []
    for k, name in zip(label_keys, label_names):
        per_class.append({
            "label": k, "name": name,
            "support": int((df["true_label"] == k).sum()),
            "false_negatives": int(((df["true_label"] == k) & (df["pred_label"] != k)).sum()),
            "false_positives": int(((df["pred_label"] == k) & (df["true_label"] != k)).sum()),
        })
    return {
        "total": int(len(df)), "errors": int(len(errors)),
        "error_rate": round(len(errors) / len(df), 4) if len(df) else None,
        "categories": [{"category": c, "count": int(n)} for c, n in errors["category"].value_counts().items()],
        "pairs": [{"true": t, "pred": p, "count": int(n)} for (t, p), n in pairs.items()],
        "per_class": per_class,
    }

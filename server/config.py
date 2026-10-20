"""
GradeSense server configuration.

Order of precedence (highest first):
  1. Environment variables GRADESENSE_<KEY> (e.g. GRADESENSE_ALLOW_CONFIG_EDIT=false) - locked in the UI
  2. config.json in the project folder (written by the Model Configuration page when editing is allowed)
  3. DEFAULTS below
"""
import copy
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
VERSION = "1.0.0"

LABEL_KEYS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]

DEFAULTS = {
    "model_dir": "models/deberta_5way",
    "model_repo": "",                 # Hugging Face repo to download the model from if model_dir is empty
    "device": "cpu",
    "max_length": 256,                # must match training (256) for faithful results
    "batch_size": 16,
    "scheme": "5way",
    "marks_mapping": {"correct": 1.0, "partially_correct_incomplete": 0.5, "contradictory": 0.0,
                      "irrelevant": 0.0, "non_domain": 0.0},
    "review_threshold": 0.6,
    "concept_covered_threshold": 0.6,
    "concept_partial_threshold": 0.45,
    "occlusion_max_words": 40,
    "show_confidence": True,
    "results_dir": "results",
    "history_enabled": True,
    "db_path": "data/gradesense.db",
    "max_question_chars": 1000,
    "max_reference_chars": 1500,
    "max_answer_chars": 1000,
    "max_batch_rows": 500,
    "max_upload_mb": 5,
    "allow_config_edit": False,       # set true in config.json for local use; keep false when public
    "profile_name": "",
    "profile_subtitle": "",
}

# key -> (type, min, max, description)
SCHEMA = {
    "model_dir": (str, None, None, "Folder with the fine-tuned model (config.json, model.safetensors, tokenizer)"),
    "model_repo": (str, None, None, "Hugging Face repo id to download the model from when model_dir is empty"),
    "device": (str, None, None, "Inference device: cpu, or cuda if a GPU is available"),
    "max_length": (int, 32, 512, "Maximum tokens per input; training used 256"),
    "batch_size": (int, 1, 64, "Answers per model pass in batch grading"),
    "scheme": (str, None, None, "Label scheme of the loaded model"),
    "marks_mapping": (dict, 0.0, 1.0, "Fraction of maximum marks awarded for each 5-way label"),
    "review_threshold": (float, 0.0, 1.0, "Flag answers whose model confidence is below this value"),
    "concept_covered_threshold": (float, 0.0, 1.0, "Sentence-BERT similarity for a concept to count as covered"),
    "concept_partial_threshold": (float, 0.0, 1.0, "Similarity for a concept to count as partially covered"),
    "occlusion_max_words": (int, 1, 120, "Longest answer (in words) that gets word-level highlighting"),
    "show_confidence": (bool, None, None, "Show the model's softmax confidence in results"),
    "results_dir": (str, None, None, "Folder with evaluation artifacts (results.csv, preds_*.csv)"),
    "history_enabled": (bool, None, None, "Store grading records in the local SQLite database"),
    "db_path": (str, None, None, "SQLite database file for grading history"),
    "max_question_chars": (int, 50, 5000, "Maximum characters in a question"),
    "max_reference_chars": (int, 50, 5000, "Maximum characters in a reference answer"),
    "max_answer_chars": (int, 20, 5000, "Maximum characters in a student answer"),
    "max_batch_rows": (int, 1, 5000, "Maximum rows per uploaded CSV"),
    "max_upload_mb": (int, 1, 50, "Maximum CSV upload size in MB"),
    "allow_config_edit": (bool, None, None, "Allow changing settings from the web interface"),
    "profile_name": (str, None, None, "Name shown in the header (leave empty to hide)"),
    "profile_subtitle": (str, None, None, "Role or institution shown under the name"),
}

# Settings that are never editable from the web interface
LOCKED_FROM_UI = {"allow_config_edit", "db_path"}

_lock = threading.Lock()


def _parse_env(key, raw):
    typ = SCHEMA[key][0]
    if typ is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    if typ is dict:
        return json.loads(raw)
    return raw


def env_overrides():
    out = {}
    for key in DEFAULTS:
        raw = os.environ.get(f"GRADESENSE_{key.upper()}")
        if raw is not None:
            try:
                out[key] = _parse_env(key, raw)
            except (ValueError, json.JSONDecodeError):
                pass
    return out


def _read_file():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if k in DEFAULTS}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def load():
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(_read_file())
    cfg.update(env_overrides())
    cfg["marks_mapping"] = {k: float(cfg["marks_mapping"].get(k, DEFAULTS["marks_mapping"][k])) for k in LABEL_KEYS}
    return cfg


def validate(updates):
    """Return (clean_updates, errors). Only known, UI-editable keys are accepted."""
    clean, errors = {}, {}
    locked = set(env_overrides()) | LOCKED_FROM_UI
    for key, value in updates.items():
        if key not in SCHEMA:
            errors[key] = "Unknown setting."
            continue
        if key in locked:
            errors[key] = "This setting is fixed by the server and cannot be changed here."
            continue
        typ, lo, hi, _ = SCHEMA[key]
        try:
            if typ is bool:
                if not isinstance(value, bool):
                    raise ValueError("must be true or false")
            elif typ is int:
                value = int(value)
            elif typ is float:
                value = float(value)
            elif typ is dict:
                if not isinstance(value, dict) or set(value) != set(LABEL_KEYS):
                    raise ValueError("must give a value for each of the five labels")
                value = {k: float(v) for k, v in value.items()}
                if any(v < lo or v > hi for v in value.values()):
                    raise ValueError(f"each value must be between {lo} and {hi}")
            else:
                value = str(value).strip()
            if typ in (int, float) and ((lo is not None and value < lo) or (hi is not None and value > hi)):
                raise ValueError(f"must be between {lo} and {hi}")
            if key == "device" and value not in ("cpu", "cuda"):
                raise ValueError("must be 'cpu' or 'cuda'")
            if key == "device" and value == "cuda":
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("no CUDA GPU is available on this machine")
            if key == "scheme" and value not in ("5way",):
                raise ValueError("the app supports the 5-way model only")
            if key in ("model_dir", "results_dir") and not value:
                raise ValueError("cannot be empty")
            clean[key] = value
        except (TypeError, ValueError) as e:
            errors[key] = str(e).capitalize() if str(e) else "Invalid value."
    merged = {**load(), **clean}
    if merged["concept_partial_threshold"] > merged["concept_covered_threshold"]:
        errors["concept_partial_threshold"] = "Must not be higher than the covered threshold."
    return clean, errors


def save(clean_updates):
    with _lock:
        data = _read_file()
        data.update(clean_updates)
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return load()


def resolve(path_value):
    p = Path(path_value)
    return p if p.is_absolute() else ROOT / p


def describe(cfg):
    locked = set(env_overrides()) | LOCKED_FROM_UI
    return [{"key": k, "value": cfg[k], "type": SCHEMA[k][0].__name__, "min": SCHEMA[k][1], "max": SCHEMA[k][2],
             "description": SCHEMA[k][3], "locked": k in locked} for k in DEFAULTS]

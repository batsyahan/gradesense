"""
GradeSense model manager: loads the fine-tuned DeBERTa grader once (in the background),
optionally downloads it from the Hugging Face Hub, loads baseline models for the Playground,
and serialises access to the model with a lock.
"""
import logging
import os
import threading
import time

from .config import resolve

log = logging.getLogger("gradesense")

PLAYGROUND_MODELS = [
    ("TFIDF-Full", "TF-IDF + LR", "Classical: word overlap and TF-IDF features"),
    ("BiLSTM", "Siamese BiLSTM", "Deep learning without pretraining"),
    ("SBERT-LR", "SBERT (No FT) + LR", "Frozen sentence embeddings + logistic regression"),
    ("DeBERTa-v3", "DeBERTa-v3 (FT)", "Fine-tuned transformer cross-encoder (the GradeSense model)"),
]


class ModelNotReady(Exception):
    pass


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.grader = None
        self.status = "not_loaded"        # not_loaded | loading | loaded | error
        self.error = None
        self.loaded_at = None
        self.load_seconds = None
        self.baselines = {}
        self.baseline_errors = {}
        self._lock = threading.Lock()      # one model pass at a time
        self._load_lock = threading.Lock()

    # ------------------------------------------------ loading
    def model_dir(self):
        return resolve(self.cfg["model_dir"])

    def has_local_model(self):
        d = self.model_dir()
        return (d / "config.json").exists() and ((d / "model.safetensors").exists()
                                                 or (d / "pytorch_model.bin").exists())

    def start_background_load(self):
        threading.Thread(target=self.load, daemon=True, name="model-loader").start()

    def load(self):
        with self._load_lock:
            if self.grader is not None:
                return
            self.status, self.error = "loading", None
            t0 = time.time()
            try:
                if not self.has_local_model():
                    repo = self.cfg.get("model_repo", "").strip()
                    if not repo:
                        raise FileNotFoundError(
                            f"No model found in '{self.cfg['model_dir']}'. Copy the fine-tuned DeBERTa files there "
                            "or set model_repo to a Hugging Face repo id.")
                    from huggingface_hub import snapshot_download
                    log.info("Downloading model from the Hugging Face Hub: %s", repo)
                    snapshot_download(repo_id=repo, local_dir=str(self.model_dir()),
                                      token=os.environ.get("HF_TOKEN") or None)
                from src.grader import Grader
                self.grader = Grader(str(self.model_dir()), device=self.cfg["device"],
                                     max_length=self.cfg["max_length"], batch_size=self.cfg["batch_size"])
                self.load_seconds = round(time.time() - t0, 1)
                self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
                self.status = "loaded"
                log.info("Model loaded in %.1fs", self.load_seconds)
            except Exception as e:  # keep the server running; the UI shows the error
                self.status, self.error = "error", str(e)
                log.exception("Model failed to load")
                return
            self._load_baselines()

    def _load_baselines(self):
        from src.baselines import BiLSTMBaseline, SbertBaseline, TfidfBaseline
        scheme = self.cfg["scheme"]
        models_dir = resolve("models")
        loaders = {
            "TFIDF-Full": lambda: TfidfBaseline.load(models_dir / f"tfidf_full_{scheme}.joblib"),
            "BiLSTM": lambda: BiLSTMBaseline.load(models_dir / f"bilstm_{scheme}.pt"),
            "SBERT-LR": lambda: SbertBaseline.load(models_dir / f"sbert_lr_{scheme}.joblib",
                                                   encoder=self.grader.sbert if self.grader else None),
        }
        for key, loader in loaders.items():
            try:
                self.baselines[key] = loader()
            except FileNotFoundError:
                self.baseline_errors[key] = "Not built yet - run: python scripts/build_baselines.py"
            except Exception as e:
                self.baseline_errors[key] = f"Could not load: {e}"

    def reload(self, cfg):
        with self._lock:
            self.cfg = cfg
            self.grader = None
            self.baselines, self.baseline_errors = {}, {}
            self.status = "not_loaded"
        self.start_background_load()

    # ------------------------------------------------ inference
    def require(self):
        if self.status != "loaded" or self.grader is None:
            if self.status == "loading" or self.status == "not_loaded":
                raise ModelNotReady("The model is still loading. Please try again in a few seconds.")
            raise ModelNotReady(f"The model is not available: {self.error}")
        return self.grader

    def analyze(self, question, reference, answer, max_marks, marks_mapping, explain):
        g = self.require()
        c = self.cfg
        with self._lock:
            return g.analyze(question, reference, answer, max_marks=max_marks, marks_mapping=marks_mapping,
                             explain=explain, review_threshold=c["review_threshold"],
                             covered_threshold=c["concept_covered_threshold"],
                             partial_threshold=c["concept_partial_threshold"],
                             occlusion_max_words=c["occlusion_max_words"])

    def grade_rows(self, questions, references, answers, max_marks, marks_mapping, with_concepts=True):
        g = self.require()
        with self._lock:
            return g.grade_rows(questions, references, answers, max_marks, marks_mapping,
                                review_threshold=self.cfg["review_threshold"], with_concepts=with_concepts)

    def playground(self, question, reference, answer):
        from src.grader import LABEL_NAMES, LABELS, VERDICT3
        g = self.require()
        out = []
        for key, name, desc in PLAYGROUND_MODELS:
            entry = {"key": key, "name": name, "description": desc, "available": False}
            try:
                t0 = time.time()
                with self._lock:
                    if key == "DeBERTa-v3":
                        p = g.probs(question, reference, [answer])[0]
                    elif key in self.baselines:
                        p = self.baselines[key].predict_proba([question], [reference], [answer])[0]
                    else:
                        entry["error"] = self.baseline_errors.get(key, "Not available")
                        out.append(entry)
                        continue
                i = int(p.argmax())
                entry.update({
                    "available": True, "label": LABELS[i], "label_name": LABEL_NAMES[LABELS[i]],
                    "verdict": VERDICT3[LABELS[i]], "confidence": round(float(p[i]), 4),
                    "probabilities": [{"label": l, "name": LABEL_NAMES[l], "p": round(float(p[k]), 4)}
                                      for k, l in enumerate(LABELS)],
                    "ms": round((time.time() - t0) * 1000),
                })
            except Exception as e:
                entry["error"] = f"Prediction failed: {e}"
            out.append(entry)
        return out

    def info(self):
        g = self.grader
        return {
            "status": self.status,
            "error": self.error,
            "loaded_at": self.loaded_at,
            "load_seconds": self.load_seconds,
            "model_dir": self.cfg["model_dir"],
            "model_repo": self.cfg["model_repo"],
            "device": self.cfg["device"],
            "max_length": self.cfg["max_length"],
            "parameters": g.n_parameters if g else None,
            "sbert": g is not None and g.sbert is not None,
            "spacy": g is not None and g.nlp is not None,
            "baselines": {key: ("loaded" if key in self.baselines else self.baseline_errors.get(key, "not loaded"))
                          for key, _, _ in PLAYGROUND_MODELS if key != "DeBERTa-v3"},
        }

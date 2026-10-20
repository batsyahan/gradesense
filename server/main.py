"""
GradeSense web server (FastAPI).

Run locally from the project folder:
    python -m uvicorn server.main:app --port 8000
then open http://localhost:8000
"""
import logging
import re
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analytics, batch
from . import config as config_mod
from .config import ROOT, VERSION
from .datasets import SPLIT_INFO, DatasetStore
from .history import History
from .inference import ModelManager, ModelNotReady

sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("gradesense")

CFG = config_mod.load()
MANAGER = ModelManager(CFG)
DATA = DatasetStore(CFG["scheme"])
HISTORY = History(CFG)

LABEL_KEYS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]
LABEL_NAMES = ["Correct", "Contradictory", "Partially Correct", "Irrelevant", "Non-Answer"]
VERDICT3 = {"correct": "Correct", "partially_correct_incomplete": "Partially Correct", "contradictory": "Incorrect",
            "irrelevant": "Incorrect", "non_domain": "Incorrect"}

EXAMPLES = [
    {"key": "correct", "name": "Correct answer",
     "question": "You used several methods to separate and identify the substances in mock rocks. "
                 "How did you separate the salt from the water?",
     "reference": "The water was evaporated, leaving the salt.",
     "answer": "We let the water evaporate and the salt stayed behind in the dish."},
    {"key": "partial", "name": "Partially correct answer",
     "question": "You used several methods to separate and identify the substances in mock rocks. "
                 "How did you separate the salt from the water?",
     "reference": "The water was evaporated, leaving the salt.",
     "answer": "Evaporate."},
    {"key": "incorrect", "name": "Incorrect answer",
     "question": "Why does a magnet stick to a car door?",
     "reference": "A car is made of a lot of iron or steel.",
     "answer": "Because magnets are fun to play with."},
    {"key": "paraphrase", "name": "Paraphrased correct answer",
     "question": "What happens to the pitch when the string is tightened?",
     "reference": "When the string is tighter, the pitch will be higher.",
     "answer": "The pitch goes up when you tighten it."},
    {"key": "confident", "name": "Confidently wrong answer",
     "question": "If the middle bulb in a series circuit burns out, what happens to the other bulbs?",
     "reference": "If the middle bulb burns out, the other 2 bulbs will go out because the circuit is broken.",
     "answer": "The other two bulbs will definitely stay lit because each bulb gets its own separate supply "
               "of electricity."},
]


def _startup():
    # Load in one thread, in order. Importing transformers fully first matters: `datasets` checks
    # transformers classes while hashing, and a half-imported module in another thread crashes it.
    try:
        import transformers
        transformers.PreTrainedTokenizerBase  # noqa: B018 - forces the lazy import to finish
    except Exception:
        pass
    DATA.load()
    MANAGER.load()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_startup, daemon=True, name="startup-loader").start()
    yield


app = FastAPI(title="GradeSense", version=VERSION, lifespan=lifespan, docs_url="/api/docs", redoc_url=None)


# ------------------------------------------------------------------ helpers
CLIENT_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def client(x_client_id):
    return x_client_id if x_client_id and CLIENT_RE.match(x_client_id) else ""


def bad(status, message):
    raise HTTPException(status_code=status, detail=message)


def _labels():
    return DATA.labels or LABEL_KEYS


@app.exception_handler(ModelNotReady)
async def not_ready_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    return {"ok": True, "model": MANAGER.status, "dataset": DATA.status}


@app.get("/api/meta")
def meta():
    c = CFG
    return {
        "version": VERSION,
        "profile": {"name": c["profile_name"], "subtitle": c["profile_subtitle"]},
        "model": MANAGER.info(),
        "dataset": {"name": "SciEntsBank", "source": "SemEval-2013 Task 7 (Hugging Face: nkazi/SciEntsBank)",
                    "scheme": c["scheme"], "status": DATA.status, "error": DATA.error},
        "labels": [{"key": k, "name": n, "verdict": VERDICT3[k]} for k, n in zip(LABEL_KEYS, LABEL_NAMES)],
        "marks_mapping": c["marks_mapping"],
        "show_confidence": c["show_confidence"],
        "limits": {"question": c["max_question_chars"], "reference": c["max_reference_chars"],
                   "answer": c["max_answer_chars"], "batch_rows": c["max_batch_rows"],
                   "upload_mb": c["max_upload_mb"]},
        "history_enabled": HISTORY.enabled,
        "config_editable": c["allow_config_edit"],
        "examples": EXAMPLES,
    }


# ------------------------------------------------------------------ grading
class GradeRequest(BaseModel):
    question: str = ""
    reference: str = ""
    answer: str = ""
    max_marks: float = 2
    marks_mapping: Optional[Dict[str, float]] = None
    explain: bool = True


def _validate_texts(question, reference, answer):
    errors = []
    for name, value, limit in (("Question", question, CFG["max_question_chars"]),
                               ("Reference answer", reference, CFG["max_reference_chars"]),
                               ("Student answer", answer, CFG["max_answer_chars"])):
        if not value.strip():
            errors.append(f"{name} is required.")
        elif len(value) > limit:
            errors.append(f"{name} is longer than {limit} characters.")
    if errors:
        bad(422, " ".join(errors))


def _mapping(m):
    if m is None:
        return CFG["marks_mapping"]
    if set(m) != set(LABEL_KEYS) or any(not 0 <= float(v) <= 1 for v in m.values()):
        bad(422, "Marks mapping must give a value between 0 and 1 for each of the five labels.")
    return m


@app.post("/api/grade")
def grade(req: GradeRequest, x_client_id: Optional[str] = Header(None)):
    _validate_texts(req.question, req.reference, req.answer)
    if not 0 < req.max_marks <= 100:
        bad(422, "Maximum marks must be between 0 and 100.")
    result = MANAGER.analyze(req.question.strip(), req.reference.strip(), req.answer.strip(), req.max_marks,
                             _mapping(req.marks_mapping), req.explain)
    cid = client(x_client_id)
    HISTORY.add_many(cid, "single", [{"question": req.question.strip(), "reference": req.reference.strip(),
                                      "answer": req.answer.strip(), "label": result["label"],
                                      "verdict": result["verdict"], "confidence": result["confidence"],
                                      "marks": result["marks"], "max_marks": result["max_marks"]}])
    result["saved_to_history"] = bool(HISTORY.enabled and cid)
    return result


class PlaygroundRequest(BaseModel):
    question: str = ""
    reference: str = ""
    answer: str = ""


@app.post("/api/playground")
def playground(req: PlaygroundRequest):
    _validate_texts(req.question, req.reference, req.answer)
    return {"models": MANAGER.playground(req.question.strip(), req.reference.strip(), req.answer.strip())}


# ------------------------------------------------------------------ dashboard + results
@app.get("/api/dashboard")
def dashboard(x_client_id: Optional[str] = Header(None)):
    scheme = CFG["scheme"]
    best = analytics.best_model(CFG, scheme)
    return {
        "history": HISTORY.stats(client(x_client_id)),
        "best_model": best,
        "deberta": analytics.split_metrics(CFG, "DeBERTa-v3", scheme, len(LABEL_KEYS)),
        "model": {"status": MANAGER.status, "error": MANAGER.error},
    }


@app.get("/api/results/comparison")
def results_comparison(metric: str = "macro_f1", scheme: str = "5way", chart_only: bool = False):
    if metric not in ("macro_f1", "weighted_f1", "accuracy"):
        bad(422, "Unknown metric.")
    return analytics.comparison(CFG, scheme, metric, chart_only)


@app.get("/api/results/confusion")
def results_confusion(model: str = "DeBERTa-v3", split: str = "test_ua", scheme: str = "5way"):
    cm = analytics.confusion(CFG, model, scheme, split, LABEL_NAMES)
    if cm is None:
        return {"available": False, "model": analytics.model_meta(model), "split": split,
                "message": f"No saved predictions for {analytics.model_meta(model)['name']} on "
                           f"{analytics.SPLIT_NAMES.get(split, split)}."}
    return {"available": True, **cm}


@app.get("/api/results/splits")
def results_splits(model: str = "DeBERTa-v3", scheme: str = "5way"):
    return analytics.split_metrics(CFG, model, scheme, len(LABEL_KEYS))


@app.get("/api/results/training")
def results_training(scheme: str = "5way"):
    h = analytics.training_history(CFG, scheme)
    return {"available": h is not None, **(h or {})}


@app.get("/api/results/status")
def results_status():
    return {"artifacts": analytics.artifact_status(CFG, MANAGER.model_dir()), "model": MANAGER.info()}


# ------------------------------------------------------------------ dataset explorer
@app.get("/api/dataset/overview")
def dataset_overview():
    try:
        return {"splits": DATA.overview(LABEL_NAMES)}
    except RuntimeError as e:
        bad(503, str(e))


@app.get("/api/dataset/samples")
def dataset_samples(split: str = "train", label: str = "", q: str = "", offset: int = Query(0, ge=0),
                    limit: int = Query(20, ge=1, le=100)):
    if split not in SPLIT_INFO:
        bad(422, "Unknown split.")
    if label and label not in LABEL_KEYS:
        bad(422, "Unknown label.")
    try:
        return DATA.samples(split, label, q.strip()[:200], offset, limit)
    except RuntimeError as e:
        bad(503, str(e))


# ------------------------------------------------------------------ error analysis
def _error_frame(model, split, scheme):
    if split not in SPLIT_INFO or split == "train":
        bad(422, "Choose a validation or test split.")
    try:
        frames = DATA.require()
    except RuntimeError as e:
        bad(503, str(e))
    df, message = analytics.error_frame(CFG, model, scheme, split, _labels(), frames)
    return df, message


def _filter_errors(df, true, pred, q, include_correct):
    if not include_correct:
        df = df[df["true_label"] != df["pred_label"]]
    if true:
        df = df[df["true_label"] == true]
    if pred:
        df = df[df["pred_label"] == pred]
    if q:
        ql = q.lower()
        df = df[df["student_answer"].fillna("").str.lower().str.contains(ql, regex=False)
                | df["question"].fillna("").str.lower().str.contains(ql, regex=False)]
    return df


@app.get("/api/errors")
def errors(model: str = "DeBERTa-v3", split: str = "test_ua", scheme: str = "5way", true: str = "",
           pred: str = "", q: str = "", include_correct: bool = False, offset: int = Query(0, ge=0),
           limit: int = Query(25, ge=1, le=100)):
    df, message = _error_frame(model, split, scheme)
    if df is None:
        return {"available": False, "message": message}
    summary = analytics.error_summary(df, LABEL_KEYS, LABEL_NAMES)
    filtered = _filter_errors(df, true, pred, q.strip()[:200], include_correct)
    page = filtered.iloc[offset:offset + limit].fillna("")
    cols = ["id", "question", "reference_answer", "student_answer", "true_label", "pred_label", "category"]
    return {"available": True, "model": analytics.model_meta(model), "split": split, "summary": summary,
            "total": int(len(filtered)), "items": page[cols].to_dict(orient="records")}


@app.get("/api/errors/export")
def errors_export(model: str = "DeBERTa-v3", split: str = "test_ua", scheme: str = "5way", true: str = "",
                  pred: str = "", q: str = "", include_correct: bool = False):
    df, message = _error_frame(model, split, scheme)
    if df is None:
        bad(404, message)
    filtered = _filter_errors(df, true, pred, q.strip()[:200], include_correct)
    cols = ["id", "question", "reference_answer", "student_answer", "true_label", "pred_label", "category"]
    name = f"errors_{model}_{split}.csv"
    return Response(filtered[cols].to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------------ batch
@app.post("/api/batch/upload")
async def batch_upload(file: UploadFile = File(...), x_client_id: Optional[str] = Header(None)):
    cid = client(x_client_id)
    if not cid:
        bad(400, "Missing client id. Reload the page and try again.")
    data = await file.read(CFG["max_upload_mb"] * 1024 * 1024 + 1)
    try:
        return batch.parse_upload(data, file.filename or "", cid, CFG)
    except batch.BatchError as e:
        bad(422, str(e))


class BatchStart(BaseModel):
    upload_id: str
    mapping: Dict[str, str]
    default_max_marks: float = 2
    with_feedback: bool = True
    marks_mapping: Optional[Dict[str, float]] = None


@app.post("/api/batch/start")
def batch_start(req: BatchStart, x_client_id: Optional[str] = Header(None)):
    cid = client(x_client_id)
    try:
        job_id = batch.start_job(req.upload_id, req.mapping, req.default_max_marks, req.with_feedback, cid, CFG,
                                 MANAGER, HISTORY, _mapping(req.marks_mapping))
    except batch.BatchError as e:
        bad(422, str(e))
    return {"job_id": job_id}


@app.get("/api/batch/jobs/{job_id}")
def batch_job(job_id: str, x_client_id: Optional[str] = Header(None)):
    try:
        job = batch.get_job(job_id, client(x_client_id))
    except batch.BatchError as e:
        bad(404, str(e))
    return batch.job_view(job, include_results=True)


@app.get("/api/batch/jobs/{job_id}/download")
def batch_download(job_id: str, x_client_id: Optional[str] = Header(None)):
    try:
        job = batch.get_job(job_id, client(x_client_id))
    except batch.BatchError as e:
        bad(404, str(e))
    if job["status"] != "completed":
        bad(409, "The batch has not finished yet.")
    return Response(batch.job_csv(job), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="graded_answers.csv"'})


# ------------------------------------------------------------------ history + search
@app.get("/api/history")
def history_list(q: str = "", offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100),
                 x_client_id: Optional[str] = Header(None)):
    return {"enabled": HISTORY.enabled, "error": HISTORY.error,
            **HISTORY.list(client(x_client_id), q.strip()[:200], offset, limit)}


@app.get("/api/history/export")
def history_export(x_client_id: Optional[str] = Header(None)):
    return Response(HISTORY.export_csv(client(x_client_id)), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="grading_history.csv"'})


@app.delete("/api/history")
def history_delete(x_client_id: Optional[str] = Header(None)):
    return {"deleted": HISTORY.delete_client(client(x_client_id))}


@app.get("/api/search")
def search(q: str = "", x_client_id: Optional[str] = Header(None)):
    q = q.strip()[:200]
    if len(q) < 2:
        return {"questions": [], "history": []}
    hist = HISTORY.list(client(x_client_id), q, 0, 5)["items"]
    return {"questions": DATA.search_questions(q),
            "history": [{"id": h["id"], "question": h["question"], "answer": h["answer"], "verdict": h["verdict"],
                         "created_at": h["created_at"]} for h in hist]}


# ------------------------------------------------------------------ configuration
@app.get("/api/config")
def get_config():
    return {"editable": CFG["allow_config_edit"], "settings": config_mod.describe(CFG), "model": MANAGER.info()}


class ConfigUpdate(BaseModel):
    updates: Dict[str, Any]


@app.put("/api/config")
def put_config(req: ConfigUpdate):
    if not CFG["allow_config_edit"]:
        bad(403, "Settings are read-only on this server.")
    clean, errors = config_mod.validate(req.updates)
    if errors:
        return JSONResponse(status_code=422, content={"detail": "Some settings are invalid.", "errors": errors})
    new_cfg = config_mod.save(clean)
    needs_reload = any(k in clean for k in ("model_dir", "model_repo", "device", "max_length", "batch_size"))
    CFG.clear()
    CFG.update(new_cfg)
    if needs_reload:
        MANAGER.reload(CFG)
    return {"saved": list(clean), "model_reloading": needs_reload,
            "note": "Inference settings changed. The model was not retrained." if needs_reload else None}


@app.post("/api/model/reload")
def reload_model():
    if not CFG["allow_config_edit"]:
        bad(403, "Reloading the model is disabled on this server.")
    MANAGER.reload(CFG)
    return {"status": "loading"}


# ------------------------------------------------------------------ frontend
app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")

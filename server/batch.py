"""
GradeSense batch grading: CSV upload -> column mapping -> background grading job with progress.
Invalid rows are reported individually and never graded; valid rows are still processed.
Uploaded files are kept in memory only (never written to disk) and expire after an hour.
"""
import io
import threading
import time
import uuid

import pandas as pd

UPLOAD_TTL = 3600
CHUNK = 16

_uploads = {}
_jobs = {}
_lock = threading.Lock()

ALIASES = {
    "question": ["question", "questions", "q", "prompt"],
    "reference": ["reference_answer", "reference", "ref", "model_answer", "expected_answer", "key"],
    "answer": ["student_answer", "answer", "response", "student_response"],
    "max_marks": ["max_marks", "marks", "max_mark", "total_marks", "out_of"],
    "student_id": ["student_id", "id", "roll_no", "roll", "usn", "name"],
}

RESULT_COLUMNS = ["verdict", "predicted_label", "model_confidence", "marks", "max_marks", "needs_review",
                  "feedback", "row_status", "row_error"]


class BatchError(Exception):
    pass


def _cleanup():
    now = time.time()
    with _lock:
        for store in (_uploads, _jobs):
            for k in [k for k, v in store.items() if now - v["created"] > UPLOAD_TTL]:
                del store[k]


def _guess(columns):
    lower = {c.lower().strip(): c for c in columns}
    mapping = {}
    for field, names in ALIASES.items():
        mapping[field] = next((lower[n] for n in names if n in lower), "")
    return mapping


def parse_upload(data, filename, client_id, cfg):
    _cleanup()
    if not filename.lower().endswith(".csv"):
        raise BatchError("Please upload a .csv file.")
    if len(data) > cfg["max_upload_mb"] * 1024 * 1024:
        raise BatchError(f"The file is larger than {cfg['max_upload_mb']} MB.")
    if not data.strip():
        raise BatchError("The file is empty.")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, skip_blank_lines=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise BatchError(f"Could not read the CSV: {str(e).splitlines()[0]}")
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) < 3:
        raise BatchError("The CSV needs at least three columns (question, reference answer, student answer).")
    if df.empty:
        raise BatchError("The CSV has a header row but no data rows.")
    if len(df) > cfg["max_batch_rows"]:
        raise BatchError(f"The CSV has {len(df)} rows; the limit is {cfg['max_batch_rows']}.")
    if len(set(df.columns)) != len(df.columns):
        raise BatchError("The CSV has duplicate column names.")
    upload_id = uuid.uuid4().hex
    with _lock:
        _uploads[upload_id] = {"df": df, "client": client_id, "created": time.time(), "filename": filename}
    return {"upload_id": upload_id, "filename": filename, "rows": int(len(df)), "columns": list(df.columns),
            "preview": df.head(8).to_dict(orient="records"), "suggested_mapping": _guess(df.columns)}


def _validate_row(row, mapping, default_max, cfg):
    q = str(row[mapping["question"]]).strip()
    r = str(row[mapping["reference"]]).strip()
    a = str(row[mapping["answer"]]).strip()
    problems = []
    if not q:
        problems.append("question is empty")
    if not r:
        problems.append("reference answer is empty")
    if not a:
        problems.append("student answer is empty")
    if len(q) > cfg["max_question_chars"]:
        problems.append(f"question longer than {cfg['max_question_chars']} characters")
    if len(r) > cfg["max_reference_chars"]:
        problems.append(f"reference answer longer than {cfg['max_reference_chars']} characters")
    if len(a) > cfg["max_answer_chars"]:
        problems.append(f"student answer longer than {cfg['max_answer_chars']} characters")
    mm = default_max
    if mapping.get("max_marks"):
        raw = str(row[mapping["max_marks"]]).strip()
        if raw:
            try:
                mm = float(raw)
                if not 0 < mm <= 100:
                    problems.append("max marks must be between 0 and 100")
            except ValueError:
                problems.append(f"max marks '{raw[:20]}' is not a number")
    return q, r, a, mm, problems


def start_job(upload_id, mapping, default_max_marks, with_feedback, client_id, cfg, manager, history,
              marks_mapping):
    with _lock:
        up = _uploads.get(upload_id)
    if up is None or up["client"] != client_id:
        raise BatchError("This upload has expired. Please upload the file again.")
    df = up["df"]
    for field in ("question", "reference", "answer"):
        col = mapping.get(field, "")
        if not col:
            raise BatchError(f"Choose a column for '{field if field != 'reference' else 'reference answer'}'.")
    for field, col in mapping.items():
        if col and col not in df.columns:
            raise BatchError(f"Column '{col}' is not in the file.")
    chosen = [mapping[f] for f in ("question", "reference", "answer")]
    if len(set(chosen)) != 3:
        raise BatchError("Question, reference answer and student answer must be three different columns.")
    if not 0 < float(default_max_marks) <= 100:
        raise BatchError("Default maximum marks must be between 0 and 100.")
    manager.require()

    job_id = uuid.uuid4().hex
    job = {"id": job_id, "client": client_id, "created": time.time(), "status": "running", "total": int(len(df)),
           "done": 0, "graded": 0, "invalid": 0, "errors": [], "results": [None] * len(df), "df": df,
           "message": None, "filename": up["filename"]}
    with _lock:
        _jobs[job_id] = job
    args = (job, mapping, float(default_max_marks), bool(with_feedback), cfg, manager, history, marks_mapping)
    threading.Thread(target=_run, args=args, daemon=True, name=f"batch-{job_id[:6]}").start()
    return job_id


def _run(job, mapping, default_max, with_feedback, cfg, manager, history, marks_mapping):
    try:
        valid = []
        for i, row in job["df"].iterrows():
            q, r, a, mm, problems = _validate_row(row, mapping, default_max, cfg)
            if problems:
                msg = "; ".join(problems)
                job["errors"].append({"row": int(i) + 2, "message": msg})   # +2: header line, 1-based rows
                job["results"][i] = {"row_status": "invalid", "row_error": msg}
                job["invalid"] += 1
                job["done"] += 1
            else:
                valid.append((i, q, r, a, mm))
        records = []
        for start in range(0, len(valid), CHUNK):
            chunk = valid[start:start + CHUNK]
            try:
                out = manager.grade_rows([c[1] for c in chunk], [c[2] for c in chunk], [c[3] for c in chunk],
                                         [c[4] for c in chunk], marks_mapping, with_concepts=with_feedback)
                for (i, q, r, a, mm), res in zip(chunk, out):
                    job["results"][i] = {**res, "row_status": "graded", "row_error": ""}
                    records.append({"question": q, "reference": r, "answer": a, "label": res["label"],
                                    "verdict": res["verdict"], "confidence": res["confidence"],
                                    "marks": res["marks"], "max_marks": res["max_marks"]})
                    job["graded"] += 1
            except Exception as e:
                for i, *_ in chunk:
                    job["results"][i] = {"row_status": "failed", "row_error": f"Grading failed: {e}"}
                    job["errors"].append({"row": int(i) + 2, "message": f"Grading failed: {e}"})
            job["done"] += len(chunk)
        if records:
            history.add_many(job["client"], "batch", records)
        job["status"] = "completed"
    except Exception as e:
        job["status"], job["message"] = "failed", f"The batch stopped unexpectedly: {e}"


def get_job(job_id, client_id):
    with _lock:
        job = _jobs.get(job_id)
    if job is None or job["client"] != client_id:
        raise BatchError("Batch job not found (it may have expired).")
    return job


def _graded_frame(job):
    df = job["df"].copy()
    rename = {c: f"{c}_original" for c in df.columns if c in RESULT_COLUMNS}
    df = df.rename(columns=rename)
    res = job["results"]
    df["verdict"] = [r.get("verdict", "") if r else "" for r in res]
    df["predicted_label"] = [r.get("label_name", "") if r else "" for r in res]
    df["model_confidence"] = [r.get("confidence", "") if r else "" for r in res]
    df["marks"] = [r.get("marks", "") if r else "" for r in res]
    df["max_marks"] = [r.get("max_marks", "") if r else "" for r in res]
    df["needs_review"] = [r.get("needs_review", "") if r else "" for r in res]
    df["feedback"] = [r.get("feedback", "") if r else "" for r in res]
    df["row_status"] = [r.get("row_status", "pending") if r else "pending" for r in res]
    df["row_error"] = [r.get("row_error", "") if r else "" for r in res]
    return df


def job_view(job, include_results=False, limit=500):
    view = {k: job[k] for k in ("id", "status", "total", "done", "graded", "invalid", "message", "filename")}
    view["progress"] = round(job["done"] / job["total"], 4) if job["total"] else 1.0
    view["errors"] = job["errors"][:200]
    if include_results and job["status"] == "completed":
        graded = [r for r in job["results"] if r and r.get("row_status") == "graded"]
        marks = [r["marks"] for r in graded]
        view["summary"] = {
            "graded": len(graded),
            "average_marks": round(sum(marks) / len(marks), 3) if marks else None,
            "average_fraction": round(sum(r["marks"] / r["max_marks"] for r in graded) / len(graded), 3)
            if graded else None,
            "verdicts": {v: sum(1 for r in graded if r["verdict"] == v)
                         for v in ("Correct", "Partially Correct", "Incorrect")},
            "labels": {l: sum(1 for r in graded if r["label_name"] == l)
                       for l in ("Correct", "Partially Correct", "Contradictory", "Irrelevant", "Non-Answer")},
            "needs_review": sum(1 for r in graded if r["needs_review"]),
            "marks_distribution": _distribution(graded),
        }
        frame = _graded_frame(job)
        view["columns"] = list(frame.columns)
        view["rows"] = frame.head(limit).fillna("").to_dict(orient="records")
    return view


def _distribution(graded):
    """Histogram of marks as a fraction of maximum marks, in 10% bins."""
    bins = [0] * 11
    for r in graded:
        frac = r["marks"] / r["max_marks"] if r["max_marks"] else 0
        bins[min(10, int(round(frac * 10)))] += 1
    return [{"bin": f"{i * 10}%", "count": c} for i, c in enumerate(bins)]


def job_csv(job):
    return _graded_frame(job).to_csv(index=False)

---
title: GradeSense
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Transformer-based short answer grading with feedback
---

# GradeSense

AI-powered short-answer grading with a fine-tuned **DeBERTa-v3** cross-encoder, concept-level feedback and word-level
explanations. Built as a Deep Learning and NLP mini-project on the SciEntsBank dataset (SemEval-2013 Task 7).

## Project layout

| Path | Contents |
|---|---|
| `server/` | FastAPI backend: API routes, model manager, analytics, dataset explorer, batch jobs, history (SQLite), configuration |
| `web/` | Frontend: plain HTML, CSS and JavaScript modules (no build step) |
| `src/` | ML code: data loading, evaluation, grading engine, baseline models |
| `scripts/build_baselines.py` | Builds the TF-IDF, BiLSTM and SBERT baseline artifacts used by the app |
| `week1_baseline.py`, `week2_bilstm.py` | Original experiment scripts |
| `results/` | Evaluation artifacts: `results.csv`, `preds_*.csv`, training history, confusion-matrix images |
| `models/` | `deberta_5way/` (fine-tuned grader, not in git) and the small baseline models |
| `deploy/deploy_space.py` | One-command deployment to Hugging Face Spaces |
| `app/app.py` | Earlier Streamlit prototype (kept for reference) |

## Run locally (Windows PowerShell)

```powershell
cd ~\GradeSense
.\venv\Scripts\Activate.ps1
pip install fastapi uvicorn python-multipart
python scripts/build_baselines.py          # once: baseline models + predictions (about 5 minutes)
python -m uvicorn server.main:app --port 8000 --no-access-log
```

Open <http://localhost:8000>. The model loads in the background (about 10-30 seconds); the sidebar shows its status.

Local settings live in `config.json` (`allow_config_edit: true` lets the Model Configuration page change server settings).

## Required artifacts

| Artifact | Created by | Needed for |
|---|---|---|
| `models/deberta_5way/` | Colab fine-tuning notebook | Grading (required) |
| `results/results.csv` | All training/evaluation scripts | Charts, tables, summary cards |
| `results/preds_deberta_5way_*.csv` | Colab evaluation cell | DeBERTa confusion matrix, precision/recall, error analysis |
| `results/history_deberta_5way.csv` | Colab training cell | Training curve |
| `models/tfidf_*`, `models/sbert_lr_*`, `results/preds_<baseline>_*` | `scripts/build_baselines.py` | Playground, baseline error analysis |
| `models/bilstm_5way.pt` | `week2_bilstm.py` | BiLSTM in the Playground and its predictions |

Anything missing is shown in the app as "Not evaluated" or with an explanation - no numbers are invented.

## Deploy to Hugging Face Spaces

1. **Upload the model to the Hub** (once, from Google Colab where the model is in Drive):
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   from huggingface_hub import login, HfApi; login()
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   src = "/content/drive/MyDrive/GradeSense/deberta_5way"
   repo = HfApi().whoami()["name"] + "/gradesense-deberta-5way"
   AutoModelForSequenceClassification.from_pretrained(src).float().push_to_hub(repo, private=False)
   AutoTokenizer.from_pretrained(src).push_to_hub(repo, private=False)
   print(repo)
   ```
2. **Deploy the app** from your laptop:
   ```powershell
   python -c "from huggingface_hub import login; login()"
   python deploy/deploy_space.py --model-repo YOUR_USERNAME/gradesense-deberta-5way --name "Your Name" --subtitle "Student | College"
   ```
3. Wait 5-10 minutes for the build, then open `https://huggingface.co/spaces/YOUR_USERNAME/gradesense`.

On the public Space, settings are read-only and the history database lives in `/tmp` (reset on restart; free Spaces
also sleep after a period without visitors). A custom domain is not part of the free tier.

## API

Interactive API documentation is served at `/api/docs`. Main endpoints: `POST /api/grade`, `POST /api/playground`,
`POST /api/batch/upload`, `POST /api/batch/start`, `GET /api/results/*`, `GET /api/dataset/*`, `GET /api/errors`,
`GET /api/history`.

## Licences and citations

SciEntsBank (CC BY 4.0, Dzikovska et al. 2013) · DeBERTa-v3 (MIT, He et al. 2021) · all-MiniLM-L6-v2 (Apache 2.0) ·
spaCy, FastAPI, Chart.js (MIT) · Lucide icons (ISC).

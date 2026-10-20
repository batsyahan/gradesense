# GradeSense - container for Hugging Face Spaces (Docker SDK) or any Docker host.
# The fine-tuned DeBERTa model is NOT baked into the image: it is downloaded at startup from the
# Hugging Face repo named in GRADESENSE_MODEL_REPO (set as a Space variable by deploy/deploy_space.py).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch first (the default PyPI wheel bundles ~2 GB of CUDA libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-server.txt .
RUN pip install -r requirements-server.txt && python -m spacy download en_core_web_sm

COPY --chown=user . .

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    GRADESENSE_ALLOW_CONFIG_EDIT=false \
    GRADESENSE_DB_PATH=/tmp/gradesense.db

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]

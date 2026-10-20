"""
Deploy GradeSense to Hugging Face Spaces (free CPU tier: 2 vCPU, 16 GB RAM).

Before running:
  1. pip install huggingface_hub
  2. Log in once:   python -c "from huggingface_hub import login; login()"   (paste a WRITE token)
  3. Upload the fine-tuned model to the Hub (see README, "Deploy" step 1).

Run from the project folder:
  python deploy/deploy_space.py --model-repo YOUR_USERNAME/gradesense-deberta-5way

What it uploads: code (server/, src/, web/, scripts/), results/ (evaluation artifacts) and the small baseline
models in models/. It never uploads venv/, data/ (history database), config.json or the DeBERTa folder -
the Space downloads DeBERTa from --model-repo at startup.
"""
import argparse
import os
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IGNORE = [
    "venv/**", ".venv/**", ".git/**", "**/__pycache__/**", "*.pyc", "data/**", "config.json",
    "models/deberta_5way/**", "notebooks/**", "app/**", ".ipynb_checkpoints/**", "*.zip",
]


def main():
    ap = argparse.ArgumentParser(description="Deploy GradeSense to a Hugging Face Space")
    ap.add_argument("--model-repo", required=True, help="Hub repo with the fine-tuned model, e.g. you/gradesense-deberta-5way")
    ap.add_argument("--space", default="gradesense", help="Space name (default: gradesense)")
    ap.add_argument("--name", default="", help="Name shown in the header (optional)")
    ap.add_argument("--subtitle", default="", help="Subtitle shown under the name (optional)")
    ap.add_argument("--private-space", action="store_true", help="Make the Space private")
    args = ap.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run:  python -c \"from huggingface_hub import login; login()\"")

    try:
        info = api.model_info(args.model_repo)
    except Exception as e:
        sys.exit(f"Cannot find model repo '{args.model_repo}': {e}")
    files = {s.rfilename for s in (info.siblings or [])}
    if not {"config.json", "model.safetensors"} <= files and not {"config.json", "pytorch_model.bin"} <= files:
        sys.exit(f"'{args.model_repo}' does not contain config.json and model weights. Upload the model first.")

    space_id = f"{user}/{args.space}"
    print(f"Creating or updating Space {space_id} …")
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True, private=args.private_space)

    print("Uploading project files (code, results, baseline models) …")
    api.upload_folder(folder_path=ROOT, repo_id=space_id, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy GradeSense")

    print("Setting Space variables …")
    api.add_space_variable(space_id, "GRADESENSE_MODEL_REPO", args.model_repo)
    if args.name:
        api.add_space_variable(space_id, "GRADESENSE_PROFILE_NAME", args.name)
    if args.subtitle:
        api.add_space_variable(space_id, "GRADESENSE_PROFILE_SUBTITLE", args.subtitle)
    if info.private:
        print("\nNOTE: the model repo is private. In the Space settings, add a secret named HF_TOKEN containing a\n"
              "READ token, or make the model repo public, otherwise the Space cannot download the model.")

    print(f"\nDone. The Space builds for about 5-10 minutes, then runs at:\n  https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()

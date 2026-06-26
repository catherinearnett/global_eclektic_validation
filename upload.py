"""
Upload model_generations_results.csv to HuggingFace dataset repo.

Usage:
    export HF_TOKEN_UPLOAD=hf_...
    python3 upload_results.py
"""

import os
from huggingface_hub import HfApi

RESULTS_PATH = "model_generations_results.csv"
REPO_ID      = "mrlbenchmarks/validation"

token = os.environ.get("HF_TOKEN_UPLOAD")
if not token:
    raise SystemExit("[ERROR] HF_TOKEN_UPLOAD environment variable not set.")

api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=RESULTS_PATH,
    path_in_repo="model_generations_results.csv",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print(f"Uploaded {RESULTS_PATH} to hf.co/datasets/{REPO_ID}")

"""
Select top 100 questions that:
1. Remove questions all models got right
2. Prioritize questions with the biggest spread in model scores

Reads from HuggingFace dataset repo using HF_TOKEN_MRL_READ.

Usage:
    export HF_TOKEN_MRL_READ=hf_...
    python3 select_top100.py
"""

import os
import io
import pandas as pd
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID    = "mrlbenchmarks/validation"
FILE_PATH  = "model_generations_results.csv"
TOKEN      = os.environ.get("HF_TOKEN_MRL_READ")

if not TOKEN:
    raise SystemExit("[ERROR] HF_TOKEN_MRL_READ environment variable not set.")

# ── Load from HF ──────────────────────────────────────────────────────────────
print("Downloading results from HuggingFace …")
api = HfApi(token=TOKEN)
content = api.hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE_PATH,
    repo_type="dataset",
)
df = pd.read_csv(content, encoding="utf-8-sig")

# Drop summary row
df = df[df["Question"] != "** MEAN EM **"].reset_index(drop=True)
print(f"  Loaded {len(df)} questions.\n")

# ── Identify EM columns ───────────────────────────────────────────────────────
em_cols = [c for c in df.columns if c.startswith("em_")]
df[em_cols] = df[em_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

# ── Filter: remove questions all models got right ─────────────────────────────
df["total_correct"] = df[em_cols].sum(axis=1)
df_filtered = df[df["total_correct"] < len(em_cols)].copy()
print(f"  After removing all-correct: {len(df_filtered)} questions remaining.")

# ── Score by spread ───────────────────────────────────────────────────────────
# Spread = variance across model scores (maximised when some get it right, some don't)
# Also weight by questions where at least one model got it right (not all wrong)
df_filtered["spread"] = df_filtered[em_cols].var(axis=1)
df_filtered["any_correct"] = df_filtered[em_cols].max(axis=1)

# Sort: prioritise spread (discriminative), then questions at least one got right
df_filtered = df_filtered.sort_values(
    by=["spread", "any_correct"],
    ascending=[False, False]
)

# ── Select top 100 ────────────────────────────────────────────────────────────
top100 = df_filtered.head(100).reset_index(drop=True)
print(f"  Selected top {len(top100)} questions by spread.\n")

print(f"  Spread distribution:")
print(f"    Max spread:  {top100['spread'].max():.4f}")
print(f"    Min spread:  {top100['spread'].min():.4f}")
print(f"    Mean spread: {top100['spread'].mean():.4f}")
print(f"    Questions with ≥1 correct: {top100['any_correct'].sum()}")

# ── Per-model average scores ─────────────────────────────────────────────────
print("\n  Per-model exact match on top 100:")


for col in em_cols:
    model_name = col.replace("em_", "")
    avg = top100[col].mean()
    print(f"    {model_name}: {avg:.2%}")

# ── Save and upload ───────────────────────────────────────────────────────────
out_cols = ["Question", "Answer"] + em_cols + ["total_correct", "spread"]
out_path = "top100_questions.csv"
top100[out_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved to {out_path}")

hf_upload_token = os.environ.get("HF_TOKEN_UPLOAD")
if hf_upload_token:
    upload_api = HfApi(token=hf_upload_token)
    upload_api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo="top100_questions.csv",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded to hf.co/datasets/{REPO_ID}")
else:
    print("[SKIP] HF_TOKEN_UPLOAD not set, skipping upload.")

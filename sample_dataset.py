"""
Select top 100 questions that:
1. Remove questions all models got right
2. Prioritize questions with the biggest spread in model scores

Sampling uses ONLY the original-language scores (em_<model>_orig).
The translated-question scores (em_<model>_en) and all dataset columns are
kept in the output, but they do not affect which questions are selected.

Reads from HuggingFace dataset repo using HF_TOKEN_MRL_READ.

Usage:
    export HF_TOKEN_MRL_READ=hf_...
    python3 select_top100.py
"""

import os
import pandas as pd
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID    = "mrlbenchmarks/validation"
FILE_PATH  = "model_generations_results.csv"
OUT_PATH   = "top100_questions.csv"
TOP_N      = 100
TOKEN      = os.environ.get("HF_TOKEN_MRL_READ")

# Which variant drives the sampling: "orig" (original language) or "en" (translation)
SAMPLE_VARIANT = "orig"

# Dataset columns to carry through to the output, in this order
KEEP_COLUMNS = [
    "ID",
    "Language",
    "Country/Region",
    "Question",
    "Answer",
    "Evidence_url",
    "Question Automatic Translation",
    "Question Corrected Translation",
    "Answer Automatic Translation",
    "Answer Corrected Translation",
    "URL language (if different from target language)",
]

if not TOKEN:
    raise SystemExit("[ERROR] HF_TOKEN_MRL_READ environment variable not set.")

# ── Load from HF ──────────────────────────────────────────────────────────────
print("Downloading results from HuggingFace …")
api = HfApi(token=TOKEN)
local_path = api.hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE_PATH,
    repo_type="dataset",
)
df = pd.read_csv(local_path, encoding="utf-8-sig", dtype={"ID": str})

# Drop summary row
df = df[df["Question"] != "** MEAN EM **"].reset_index(drop=True)
print(f"  Loaded {len(df)} questions.\n")

# ── Identify columns ──────────────────────────────────────────────────────────
dataset_cols = [c for c in KEEP_COLUMNS if c in df.columns]
missing_cols = [c for c in KEEP_COLUMNS if c not in df.columns]
if missing_cols:
    print(f"  [INFO] Not in results file, skipped: {missing_cols}")

gen_cols    = [c for c in df.columns if c.startswith("gen_")]
em_cols_all = [c for c in df.columns if c.startswith("em_")]
em_cols     = [c for c in em_cols_all if c.endswith(f"_{SAMPLE_VARIANT}")]

if not em_cols:
    raise SystemExit(
        f"[ERROR] No em_*_{SAMPLE_VARIANT} columns found. "
        f"EM columns present: {em_cols_all}"
    )
print(f"  Sampling on {len(em_cols)} '{SAMPLE_VARIANT}' columns: {em_cols}\n")

df[em_cols_all] = (
    df[em_cols_all].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
)

# ── Count correct models per variant (saved for both, sampling uses one) ─────
for variant in ("orig", "en"):
    variant_cols = [c for c in em_cols_all if c.endswith(f"_{variant}")]
    df[f"total_correct_{variant}"] = df[variant_cols].sum(axis=1)

sample_total_col = f"total_correct_{SAMPLE_VARIANT}"

# ── Filter: remove questions all models got right (sampling variant only) ────
df_filtered = df[df[sample_total_col] < len(em_cols)].copy()
print(f"  After removing all-correct: {len(df_filtered)} questions remaining.")

# ── Score by spread ───────────────────────────────────────────────────────────
# Spread = variance across model scores (maximised when some get it right, some don't)
df_filtered["spread"] = df_filtered[em_cols].var(axis=1)
df_filtered["any_correct"] = df_filtered[em_cols].max(axis=1)

# Sort: prioritise spread (discriminative), then questions at least one got right
df_filtered = df_filtered.sort_values(
    by=["spread", "any_correct"],
    ascending=[False, False],
)

# ── Select top N ──────────────────────────────────────────────────────────────
top = df_filtered.head(TOP_N).reset_index(drop=True)
print(f"  Selected top {len(top)} questions by spread.\n")

print("  Spread distribution:")
print(f"    Max spread:  {top['spread'].max():.4f}")
print(f"    Min spread:  {top['spread'].min():.4f}")
print(f"    Mean spread: {top['spread'].mean():.4f}")
print(f"    Questions with ≥1 correct: {top['any_correct'].sum()}")

# ── Per-model average scores ─────────────────────────────────────────────────
print(f"\n  Per-model exact match on top {len(top)}:")
for col in em_cols_all:
    tag = " (used for sampling)" if col in em_cols else ""
    print(f"    {col.replace('em_', '')}: {top[col].mean():.2%}{tag}")

# ── Save and upload ───────────────────────────────────────────────────────────
out_cols = dataset_cols + gen_cols + em_cols_all + [
    "total_correct_orig", "total_correct_en", "spread",
]
top[out_cols].to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved to {OUT_PATH}")

hf_upload_token = os.environ.get("HF_TOKEN_UPLOAD")
if hf_upload_token:
    upload_api = HfApi(token=hf_upload_token)
    upload_api.upload_file(
        path_or_fileobj=OUT_PATH,
        path_in_repo="top100_questions.csv",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded to hf.co/datasets/{REPO_ID}")
else:
    print("[SKIP] HF_TOKEN_UPLOAD not set, skipping upload.")

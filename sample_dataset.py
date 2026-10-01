"""
Select the top 100 questions per language that:
1. Remove questions all models got right
2. Prioritize questions with the biggest spread in model scores

Loops over every language's results from model_generations.py.
Sampling uses ONLY the original-language scores (em_<model>_orig).
The translated-question scores (em_<model>_en) and all dataset columns are
kept in the output, but they do not affect which questions are selected.

Output: top100/<language>.csv, one file per language. Upload with upload.py.

Usage:
    python3 select_top100.py                     # read results/ written by model_generations.py
    RESULTS_SOURCE=hf python3 select_top100.py   # read the generations uploaded to the HF repo
                                                 # (needs HF_TOKEN_MRL_READ)
"""

import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
# Where to read each language's results from:
#   "local": results/<language>.csv written by model_generations.py
#   "hf":    data/generations/<language>.csv in REPO_ID (uploaded by upload.py)
RESULTS_SOURCE    = os.environ.get("RESULTS_SOURCE", "local")
LOCAL_RESULTS_DIR = "results"
REPO_ID           = "mrlbenchmarks/global_eclektic"
HF_RESULTS_DIR    = "data/generations"   # must match upload.py

OUT_DIR = "top100"
TOP_N   = 100

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
# Note: "Exclude" is not listed here; model_generations.py removes excluded
# rows and drops that column before writing the results file.


# ── Find and load each language's results ────────────────────────────────────
def list_languages() -> dict:
    """Map language -> CSV path (local) or repo filename (hf)."""
    if RESULTS_SOURCE == "local":
        if not os.path.isdir(LOCAL_RESULTS_DIR):
            raise SystemExit(f"[ERROR] {LOCAL_RESULTS_DIR}/ not found; run model_generations.py first.")
        return {
            os.path.splitext(f)[0]: os.path.join(LOCAL_RESULTS_DIR, f)
            for f in sorted(os.listdir(LOCAL_RESULTS_DIR)) if f.endswith(".csv")
        }
    if RESULTS_SOURCE == "hf":
        from huggingface_hub import HfApi
        files = HfApi(token=hf_token()).list_repo_files(REPO_ID, repo_type="dataset")
        return {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in sorted(files)
            if f.startswith(HF_RESULTS_DIR + "/") and f.endswith(".csv")
        }
    raise SystemExit(f"[ERROR] RESULTS_SOURCE must be 'local' or 'hf', not '{RESULTS_SOURCE}'.")


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN_MRL_READ")
    if not token:
        raise SystemExit("[ERROR] HF_TOKEN_MRL_READ environment variable not set.")
    return token


def load_results(location: str) -> pd.DataFrame:
    if RESULTS_SOURCE == "hf":
        from huggingface_hub import hf_hub_download
        location = hf_hub_download(REPO_ID, filename=location, repo_type="dataset",
                                   token=hf_token())
    df = pd.read_csv(location, encoding="utf-8-sig", dtype={"ID": str})
    # Drop summary row
    return df[df["Question"] != "** MEAN EM **"].reset_index(drop=True)


# ── Selection for one language ────────────────────────────────────────────────
def select_top(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    print(f"\n── {lang} " + "─" * max(0, 60 - len(lang)))
    print(f"  Loaded {len(df)} questions.")

    dataset_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    missing_cols = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  [INFO] Not in results file, skipped: {missing_cols}")

    gen_cols    = [c for c in df.columns if c.startswith("gen_")]
    em_cols_all = [c for c in df.columns if c.startswith("em_")]
    em_cols     = [c for c in em_cols_all if c.endswith(f"_{SAMPLE_VARIANT}")]

    if not em_cols:
        print(f"  [SKIP] No em_*_{SAMPLE_VARIANT} columns found. EM columns present: {em_cols_all}")
        return None
    print(f"  Sampling on {len(em_cols)} '{SAMPLE_VARIANT}' columns.")

    df[em_cols_all] = (
        df[em_cols_all].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )

    # Count correct models per variant (saved for both, sampling uses one)
    for variant in ("orig", "en"):
        variant_cols = [c for c in em_cols_all if c.endswith(f"_{variant}")]
        df[f"total_correct_{variant}"] = df[variant_cols].sum(axis=1)

    sample_total_col = f"total_correct_{SAMPLE_VARIANT}"

    # Filter: remove questions all models got right (sampling variant only)
    df_filtered = df[df[sample_total_col] < len(em_cols)].copy()
    print(f"  After removing all-correct: {len(df_filtered)} questions remaining.")

    # Spread = variance across model scores (maximised when some get it right, some don't)
    df_filtered["spread"] = df_filtered[em_cols].var(axis=1)
    df_filtered["any_correct"] = df_filtered[em_cols].max(axis=1)

    # Sort: prioritise spread (discriminative), then questions at least one got right
    df_filtered = df_filtered.sort_values(
        by=["spread", "any_correct"],
        ascending=[False, False],
    )

    top = df_filtered.head(TOP_N).reset_index(drop=True)
    print(f"  Selected top {len(top)} questions by spread.")
    if len(top) < TOP_N:
        print(f"  [INFO] Fewer than {TOP_N} questions available for {lang}.")

    if len(top):
        print("  Spread distribution:")
        print(f"    Max spread:  {top['spread'].max():.4f}")
        print(f"    Min spread:  {top['spread'].min():.4f}")
        print(f"    Mean spread: {top['spread'].mean():.4f}")
        print(f"    Questions with ≥1 correct: {top['any_correct'].sum()}")

        print(f"  Per-model exact match on top {len(top)}:")
        for col in em_cols_all:
            tag = " (used for sampling)" if col in em_cols else ""
            print(f"    {col.replace('em_', '')}: {top[col].mean():.2%}{tag}")

    out_cols = dataset_cols + gen_cols + em_cols_all + [
        "total_correct_orig", "total_correct_en", "spread",
    ]
    return top[out_cols]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    languages = list_languages()
    if not languages:
        raise SystemExit("[ERROR] No language results found.")
    print(f"Found {len(languages)} languages ({RESULTS_SOURCE}): {', '.join(languages)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    saved = []
    for lang, location in languages.items():
        top = select_top(load_results(location), lang)
        if top is None:
            continue
        out_path = os.path.join(OUT_DIR, f"{lang}.csv")
        top.to_csv(out_path, index=False, encoding="utf-8-sig")
        saved.append(out_path)
        print(f"  Saved to {out_path}")

    print(f"\nSaved {len(saved)} files to {OUT_DIR}/. Upload with:  python3 upload.py")


if __name__ == "__main__":
    main()

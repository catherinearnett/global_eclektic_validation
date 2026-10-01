"""
Generate answers from multiple models locally on GPU using transformers,
and evaluate using exact match scoring.

Each model is queried twice per row:
  - "orig": the original-language question  (Question -> Answer)
  - "en":   the English translation          (Question Corrected Translation -> Answer Corrected Translation)

Requirements:
    pip install transformers accelerate pandas torch

Usage:
    python3 model_generations.py
"""

import re
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "global_eclektic_unfiltered/Ukrainian - Questions.csv"
OUT_PATH = "model_generations_results.csv"

MODELS = [
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-27B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "swiss-ai/Apertus-70B-Instruct-2509",
]

MAX_NEW_TOKENS = 100

# Columns to keep, in this order
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
    "Exclude",
]

# (automatic, corrected) translation column pairs
TRANSLATION_PAIRS = [
    ("Question Automatic Translation", "Question Corrected Translation"),
    ("Answer Automatic Translation", "Answer Corrected Translation"),
]

# Question/answer variants to evaluate: name -> (question column, gold answer column)
VARIANTS = {
    "orig": ("Question", "Answer"),
    "en": ("Question Corrected Translation", "Answer Corrected Translation"),
}


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df)} raw rows.\n")
    return df


# ── Filter dataset ────────────────────────────────────────────────────────────
def _is_blank(series: pd.Series) -> pd.Series:
    """True where a cell is NaN or only whitespace."""
    return series.isna() | series.astype(str).str.strip().eq("")


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    # Keep only the expected columns, in the expected order
    df = df[KEEP_COLUMNS].copy()
    df = df.apply(lambda s: s.str.strip())
    n_start = len(df)

    # Drop any row with a value in Exclude, then drop the column
    excluded = ~_is_blank(df["Exclude"])
    df = df[~excluded].drop(columns="Exclude")
    print(f"  Removed {excluded.sum()} excluded rows.")

    # Fill empty corrected translations from the automatic translation
    for auto_col, corr_col in TRANSLATION_PAIRS:
        fill = _is_blank(df[corr_col])
        df.loc[fill, corr_col] = df.loc[fill, auto_col]
        print(f"  Filled {fill.sum()} empty '{corr_col}' from '{auto_col}'.")

    # Drop rows still missing a translation (no automatic AND no corrected)
    no_translation = pd.Series(False, index=df.index)
    for _, corr_col in TRANSLATION_PAIRS:
        no_translation |= _is_blank(df[corr_col])
    df = df[~no_translation]
    print(f"  Removed {no_translation.sum()} rows with no translation.")

    # Drop rows missing the original question or answer
    no_original = _is_blank(df["Question"]) | _is_blank(df["Answer"])
    df = df[~no_original]
    print(f"  Removed {no_original.sum()} rows with no original question/answer.")

    df = df.reset_index(drop=True)
    print(f"  Kept {len(df)} of {n_start} rows.\n")
    return df


# ── Inference ─────────────────────────────────────────────────────────────────
def load_model(model_id: str):
    print(f"  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"  Loading model across GPUs …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",        # spreads across all available GPUs
    )
    model.eval()
    return tokenizer, model


def query_model(tokenizer, model, question: str, model_id: str = "") -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the following question as briefly as possible. "
                "Give only the answer, no explanation."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Qwen3 has thinking mode on by default — disable it
    is_qwen3 = "Qwen3" in model_id or "Qwen3" in type(tokenizer).__name__
    template_kwargs = {"enable_thinking": False} if is_qwen3 else {}

    try:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    except Exception as e:
        import traceback
        print(f"    [WARN] Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return ""


# ── Scoring ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))


def write_model_columns(results: pd.DataFrame, df: pd.DataFrame,
                        safe_name: str, generations: dict) -> None:
    """Write gen/em columns for every variant, padding rows not yet generated."""
    n = len(df)
    for variant, (_, answer_col) in VARIANTS.items():
        gens = generations[variant]
        pad = n - len(gens)
        results[f"gen_{safe_name}_{variant}"] = gens + [""] * pad
        results[f"em_{safe_name}_{variant}"] = [
            exact_match(g, gold) for g, gold in zip(gens, df[answer_col])
        ] + [None] * pad


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = filter_dataset(load_dataset(CSV_PATH))
    results = df[["ID", "Question", "Answer",
                  "Question Corrected Translation",
                  "Answer Corrected Translation"]].copy()

    for model_id in MODELS:
        safe_name = model_id.split("/")[-1]

        print(f"\n{'='*60}")
        print(f"Querying {model_id} …")
        print(f"{'='*60}")

        tokenizer, model = load_model(model_id)

        generations = {variant: [] for variant in VARIANTS}
        for i, row in df.iterrows():
            for variant, (question_col, _) in VARIANTS.items():
                question = row[question_col]
                print(f"  [{i+1}/{len(df)}][{variant}] {question[:60]}")
                gen = query_model(tokenizer, model, question, model_id)
                print(f"         → {gen[:60]}")
                generations[variant].append(gen)

            # Save incrementally every 10 rows
            if (i + 1) % 10 == 0:
                write_model_columns(results, df, safe_name, generations)
                results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
                print(f"    [checkpoint] saved after {i+1} rows")

        write_model_columns(results, df, safe_name, generations)
        for variant in VARIANTS:
            avg_em = results[f"em_{safe_name}_{variant}"].mean()
            print(f"\n  → Exact Match for {safe_name} [{variant}]: {avg_em:.2%}")
        results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"  [saved] {safe_name} complete")

        # Free GPU memory before loading next model
        del model, tokenizer
        torch.cuda.empty_cache()

    # ── Summary row ──────────────────────────────────────────────────────────
    em_cols = [c for c in results.columns if c.startswith("em_")]

    # Cast EM columns to int before summary row
    for col in em_cols:
        results[col] = results[col].astype(int)

    summary = {"ID": "", "Question": "** MEAN EM **", "Answer": ""}
    for col in em_cols:
        summary[col] = round(results[col].mean(), 4)
    results = pd.concat([results, pd.DataFrame([summary])], ignore_index=True)

    # ── Save locally ─────────────────────────────────────────────────────────
    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {OUT_PATH}")
    print(results.to_string(max_colwidth=50))

    # ── Upload to HuggingFace dataset repo ───────────────────────────────────
    hf_upload_token = os.environ.get("HF_TOKEN_UPLOAD")
    if hf_upload_token:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_upload_token)
        api.upload_file(
            path_or_fileobj=OUT_PATH,
            path_in_repo="model_generations_results.csv",
            repo_id="mrlbenchmarks/validation",
            repo_type="dataset",
        )
        print("Uploaded to hf.co/datasets/mrlbenchmarks/validation")
    else:
        print("[SKIP] HF_TOKEN_UPLOAD not set, skipping upload.")

    return results


if __name__ == "__main__":
    main()

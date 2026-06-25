"""
Generate answers from multiple HuggingFace models on a multilingual QA dataset
and evaluate using exact match scoring.

Requirements:
    pip install huggingface_hub pandas

Usage:
    python3 model_generations.py
"""

import re
import pandas as pd
from huggingface_hub import InferenceClient

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/ukr_test.csv"

MODELS = [
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-27B",           
    "swiss-ai/Apertus-70B-Instruct-2509",
]

MAX_NEW_TOKENS = 50


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the QA dataset from a local CSV file."""
    print(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path)
    # Keep only rows where both Question and Answer are present
    df = df[df["Question"].notna() & df["Answer"].notna()].reset_index(drop=True)
    print(f"  Loaded {len(df)} rows.\n")
    return df


# ── Inference ─────────────────────────────────────────────────────────────────
def build_prompt(question: str) -> str:
    """Wrap a question in a minimal zero-shot prompt."""
    return (
        "Answer the following question as briefly as possible. "
        "Give only the answer, no explanation.\n\n"
        f"Question: {question}\nAnswer:"
    )


def query_model(client: InferenceClient, model: str, question: str) -> str:
    """Call the HF Inference API and return the generated answer."""
    prompt = build_prompt(question)
    try:
        response = client.text_generation(
            prompt,
            model=model,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            stop_sequences=["\n", "<|end|>", "<|eot_id|>"],
        )
        return response.strip()
    except Exception as e:
        print(f"    [WARN] {model} failed for question '{question[:40]}…': {e}")
        return ""


# ── Scoring ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lower-case, strip punctuation/whitespace for lenient exact match."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    return " ".join(text.split())         # collapse whitespace


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 if normalised prediction equals normalised gold, else 0."""
    return int(normalize(prediction) == normalize(gold))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_dataset(CSV_PATH)

    client = InferenceClient()  # uses free serverless Inference API (no token needed)

    results = df[["Question", "Answer"]].copy()

    for model in MODELS:
        safe_name = model.split("/")[-1]
        gen_col   = f"gen_{safe_name}"
        em_col    = f"em_{safe_name}"

        print(f"Querying {model} …")
        generations = []
        for i, row in df.iterrows():
            question = row["Question"]
            print(f"  [{i+1}/{len(df)}] {question[:60]}")
            gen = query_model(client, model, question)
            generations.append(gen)

        results[gen_col] = generations
        results[em_col]  = [
            exact_match(gen, gold)
            for gen, gold in zip(generations, df["Answer"])
        ]

        avg_em = results[em_col].mean()
        print(f"  → Exact Match for {safe_name}: {avg_em:.2%}\n")

    # ── Summary row ──────────────────────────────────────────────────────────
    em_cols = [c for c in results.columns if c.startswith("em_")]
    summary = {"Question": "** MEAN EM **", "Answer": ""}
    for col in em_cols:
        summary[col] = results[col].mean()
    results = pd.concat([results, pd.DataFrame([summary])], ignore_index=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = "model_generations_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    print(results.to_string(max_colwidth=50))
    return results


if __name__ == "__main__":
    main()

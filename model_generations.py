"""
Generate answers from multiple models locally on GPU using transformers,
and evaluate using exact match scoring.

Requirements:
    pip install transformers accelerate pandas torch

Usage:
    python3 model_generations.py
"""

import re
import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/ukr_test.csv"

MODELS = [
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-27B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "swiss-ai/Apertus-70B-Instruct-2509",
]

MAX_NEW_TOKENS = 50


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "rb") as f:
    print(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df[df["Question"].notna() & df["Answer"].notna()].reset_index(drop=True)
    print(f"  Loaded {len(df)} rows.\n")
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_dataset(CSV_PATH)
    results = df[["Question", "Answer"]].copy()

    for model_id in MODELS:
        safe_name = model_id.split("/")[-1]
        gen_col   = f"gen_{safe_name}"
        em_col    = f"em_{safe_name}"

        print(f"\n{'='*60}")
        print(f"Querying {model_id} …")
        print(f"{'='*60}")

        tokenizer, model = load_model(model_id)

        generations = []
        for i, row in df.iterrows():
            question = row["Question"]
            print(f"  [{i+1}/{len(df)}] {question[:60]}")
            gen = query_model(tokenizer, model, question, model_id)
            print(f"         → {gen[:60]}")
            generations.append(gen)

            # Save incrementally every 10 questions
            if (i + 1) % 10 == 0:
                results[gen_col] = generations + [""] * (len(df) - len(generations))
                results[em_col]  = [
                    exact_match(g, gold)
                    for g, gold in zip(generations, df["Answer"])
                ] + [None] * (len(df) - len(generations))
                results.to_csv("model_generations_results.csv", index=False, encoding="utf-8-sig")
                print(f"    [checkpoint] saved after {i+1} questions")

        results[gen_col] = generations
        results[em_col]  = [
            exact_match(gen, gold)
            for gen, gold in zip(generations, df["Answer"])
        ]

        avg_em = results[em_col].mean()
        print(f"\n  → Exact Match for {safe_name}: {avg_em:.2%}")
        results.to_csv("model_generations_results.csv", index=False, encoding="utf-8-sig")
        print(f"  [saved] {safe_name} complete")

        # Free GPU memory before loading next model
        del model, tokenizer
        torch.cuda.empty_cache()

    # ── Summary row ──────────────────────────────────────────────────────────
    em_cols = [c for c in results.columns if c.startswith("em_")]

    # Cast EM columns to int before summary row
    for col in em_cols:
        results[col] = results[col].astype(int)

    summary = {"Question": "** MEAN EM **", "Answer": ""}
    for col in em_cols:
        summary[col] = round(results[col].mean(), 4)
    results = pd.concat([results, pd.DataFrame([summary])], ignore_index=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = "model_generations_results.csv"
    results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {out_path}")
    print(results.to_string(max_colwidth=50))
    return results


if __name__ == "__main__":
    main()

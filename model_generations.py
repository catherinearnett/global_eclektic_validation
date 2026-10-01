"""
Generate answers from multiple models locally on GPU using vLLM,
and evaluate using exact match scoring.

Each model is queried twice per row:
  - "orig": the original-language question  (Question -> Answer)
  - "en":   the English translation          (Question Corrected Translation -> Answer Corrected Translation)

Models run in parallel, each in its own process on its own set of GPUs
(see GPUS_PER_MODEL). When a model finishes, its GPUs are freed and the next
waiting model starts.

Requirements:
    pip install vllm pandas

Usage:
    export HF_TOKEN_READ=hf_...             # token with access to the gated models
    python3 model_generations.py            # run everything
    python3 model_generations.py --rerun    # ignore finished per-model results and regenerate
"""

import argparse
import os
import re
import subprocess
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

# Hugging Face libraries (vLLM, transformers) only read HF_TOKEN, so copy the
# read token into it. Needed for gated models such as Llama.
if os.environ.get("HF_TOKEN_READ"):
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN_READ"]

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "global_eclektic_unfiltered/Ukrainian - Questions.csv"
OUT_PATH = "model_generations_results.csv"
WORK_DIR = "work"          # filtered dataset + per-model results
LOG_DIR  = "logs"          # one log file per model

# Model -> number of GPUs (tensor parallel size).
# Must be a power of 2 that divides the model's attention heads (1, 2, 4, 8).
GPUS_PER_MODEL = {
    "google/gemma-4-31B-it":              2,
    "Qwen/Qwen3.6-27B":                   2,
    "meta-llama/Llama-3.3-70B-Instruct":  4,
    "swiss-ai/Apertus-70B-Instruct-2509": 4,
}
MODELS = list(GPUS_PER_MODEL)

TOTAL_GPUS = None          # None = detect automatically

MAX_NEW_TOKENS = 50
MAX_MODEL_LEN  = 4096      # prompts are short; a small context leaves more memory for batching
GPU_MEM_UTIL   = 0.90

SYSTEM_PROMPT = (
    "Answer the following question as briefly as possible. "
    "Give only the answer, no explanation."
)

# Columns to keep, in this order
KEEP_COLUMNS = [
    "ID",
    "Author",
    "Checked By",
    "Language",
    "Country/Region",
    "Question",
    "Answer",
    "Evidence_url",
    "Question Automatic Translation",
    "Question Corrected Translation",
    "Answer Automatic Translation",
    "Answer Corrected Translation",
    "Translation Corrected By",
    "Notes",
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

FILTERED_PATH = os.path.join(WORK_DIR, "filtered_dataset.csv")


def safe_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def model_result_path(model_id: str) -> str:
    return os.path.join(WORK_DIR, f"gen_{safe_name(model_id)}.csv")


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


def read_filtered() -> pd.DataFrame:
    return pd.read_csv(FILTERED_PATH, encoding="utf-8", dtype=str, keep_default_na=False)


# ── Scoring ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))


# ── Worker: one model, run in its own process on its own GPUs ────────────────
def build_prompt_ids(tokenizer, question: str, model_id: str) -> list:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # Qwen3 has thinking mode on by default — disable it
    template_kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **template_kwargs
    )
    # The chat template already contains BOS, so don't add special tokens again
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def run_worker(model_id: str, n_gpus: int, out_path: str) -> None:
    from vllm import LLM, SamplingParams

    df = read_filtered()
    name = safe_name(model_id)
    print(f"[{name}] {len(df)} rows, tensor_parallel_size={n_gpus}", flush=True)

    llm = LLM(
        model=model_id,
        tensor_parallel_size=n_gpus,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    # Build every prompt for every variant and generate in a single batched call
    prompts, keys = [], []
    for variant, (question_col, _) in VARIANTS.items():
        for q in df[question_col]:
            prompts.append({"prompt_token_ids": build_prompt_ids(tokenizer, q, model_id)})
            keys.append(variant)

    start = time.time()
    outputs = llm.generate(prompts, params)
    print(f"[{name}] generated {len(outputs)} answers in {time.time() - start:.0f}s", flush=True)

    gens = {variant: [] for variant in VARIANTS}
    for variant, out in zip(keys, outputs):
        gens[variant].append(out.outputs[0].text.strip())

    result = pd.DataFrame({"ID": df["ID"]})
    for variant, (_, answer_col) in VARIANTS.items():
        result[f"gen_{name}_{variant}"] = gens[variant]
        result[f"em_{name}_{variant}"] = [
            exact_match(g, gold) for g, gold in zip(gens[variant], df[answer_col])
        ]
        print(f"[{name}] Exact Match [{variant}]: "
              f"{result[f'em_{name}_{variant}'].mean():.2%}", flush=True)

    tmp_path = out_path + ".tmp"
    result.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, out_path)   # only appears once complete
    print(f"[{name}] saved {out_path}", flush=True)


# ── Scheduler: run workers in parallel on separate GPUs ──────────────────────
def detect_gpus() -> int:
    if TOTAL_GPUS:
        return TOTAL_GPUS
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
        return len([line for line in out.splitlines() if line.strip()])
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("[ERROR] Could not detect GPUs; set TOTAL_GPUS in the config.")


def is_done(model_id: str, df: pd.DataFrame) -> bool:
    """A model is done if its result file exists and matches the current dataset."""
    path = model_result_path(model_id)
    if not os.path.exists(path):
        return False
    prev = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str}, keep_default_na=False)
    return prev["ID"].tolist() == df["ID"].tolist()


def run_all_models(df: pd.DataFrame, rerun: bool) -> list:
    os.makedirs(LOG_DIR, exist_ok=True)
    total = detect_gpus()
    print(f"Detected {total} GPUs.\n")

    pending = []
    for model_id in MODELS:
        if GPUS_PER_MODEL[model_id] > total:
            raise SystemExit(f"[ERROR] {model_id} needs {GPUS_PER_MODEL[model_id]} GPUs, "
                             f"only {total} available.")
        if not rerun and is_done(model_id, df):
            print(f"  [skip] {safe_name(model_id)} already has results "
                  f"(use --rerun to regenerate)")
        else:
            pending.append(model_id)

    free_gpus = list(range(total))
    running = {}   # model_id -> (Popen, gpu list, log file, start time)
    failed = []

    while pending or running:
        # Start every pending model that fits on the free GPUs, in MODELS order
        for model_id in list(pending):
            n = GPUS_PER_MODEL[model_id]
            if n <= len(free_gpus):
                gpus, free_gpus = free_gpus[:n], free_gpus[n:]
                log_path = os.path.join(LOG_DIR, f"{safe_name(model_id)}.log")
                log = open(log_path, "w", encoding="utf-8")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
                proc = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__),
                     "--worker", model_id,
                     "--gpus", str(n),
                     "--out", model_result_path(model_id)],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                running[model_id] = (proc, gpus, log, time.time())
                pending.remove(model_id)
                print(f"  [start] {safe_name(model_id)} on GPUs {gpus}  (log: {log_path})")

        # Check for finished models and free their GPUs
        for model_id, (proc, gpus, log, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            free_gpus = sorted(free_gpus + gpus)
            del running[model_id]
            mins = (time.time() - started) / 60
            if code == 0:
                print(f"  [done]  {safe_name(model_id)} in {mins:.1f} min, freed GPUs {gpus}")
            else:
                failed.append(model_id)
                log_path = os.path.join(LOG_DIR, f"{safe_name(model_id)}.log")
                print(f"  [FAIL]  {safe_name(model_id)} exited with code {code} "
                      f"after {mins:.1f} min, see {log_path}")
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
                # Show the actual error messages (root cause), not just the traceback tail
                error_lines = []
                for line in lines:
                    text = re.sub(r"^\([^)]*\)\s*", "", line.strip())  # drop "(EngineCore pid=…)" prefix
                    if re.match(r"^(\S+\.)?\w*(Error|Exception)\w*:", text) and text not in error_lines:
                        error_lines.append(text)
                shown = error_lines[-8:] if error_lines else lines[-15:]
                print("          " + "\n          ".join(shown))

        time.sleep(5)

    return failed


# ── Merge per-model results into the final file ───────────────────────────────
def merge_results(df: pd.DataFrame) -> pd.DataFrame:
    # Keep every dataset column except Exclude (already applied in filter_dataset)
    results = df.drop(columns="Exclude", errors="ignore").copy()

    for model_id in MODELS:
        if not is_done(model_id, df):
            print(f"  [WARN] No results for {safe_name(model_id)}, left out of the final file.")
            continue
        model_df = pd.read_csv(model_result_path(model_id), encoding="utf-8-sig",
                               dtype={"ID": str}, keep_default_na=False)
        results = pd.concat([results, model_df.drop(columns="ID")], axis=1)

    em_cols = [c for c in results.columns if c.startswith("em_")]
    for col in em_cols:
        results[col] = results[col].astype(int)

    print("\nExact Match per model:")
    for col in em_cols:
        print(f"  {col.replace('em_', '')}: {results[col].mean():.2%}")

    # Summary row
    summary = {"ID": "", "Question": "** MEAN EM **", "Answer": ""}
    for col in em_cols:
        summary[col] = round(results[col].mean(), 4)
    return pd.concat([results, pd.DataFrame([summary])], ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="regenerate models that already have results")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.gpus, args.out)
        return

    # Fail fast if vLLM isn't installed in this Python (workers use the same one)
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            f"[ERROR] vLLM is not installed for {sys.executable}\n"
            f"        Install it with:  {sys.executable} -m pip install -U vllm"
        )

    if not os.environ.get("HF_TOKEN"):
        print("[WARN] HF_TOKEN_READ is not set; gated models (e.g. Llama) will fail to download.\n")

    os.makedirs(WORK_DIR, exist_ok=True)
    df = filter_dataset(load_dataset(CSV_PATH))
    df.to_csv(FILTERED_PATH, index=False, encoding="utf-8")
    df = read_filtered()   # read back so workers and merge see identical data

    failed = run_all_models(df, args.rerun)
    if failed:
        print(f"\n[WARN] Failed models: {[safe_name(m) for m in failed]}")

    results = merge_results(df)
    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {OUT_PATH}")

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


if __name__ == "__main__":
    main()

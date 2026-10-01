"""
Generate answers from multiple models locally on GPU using vLLM,
and evaluate using exact match scoring.

Runs every language file in DATA_DIR (e.g. "Ukrainian - Questions.csv").
Each model is loaded once and answers every language in one batched call.

Each model is queried twice per row:
  - "orig": the original-language question  (Question -> Answer)
  - "en":   the English translation          (Question Corrected Translation -> Answer Corrected Translation)

Models run in parallel, each in its own process on its own set of GPUs
(see GPUS_PER_MODEL). When a model finishes, its GPUs are freed and the next
waiting model starts.

Output: results/<language>.csv, one file per language (e.g. results/ukrainian.csv).
Upload them with upload.py.

Requirements:
    pip install vllm pandas

Usage:
    export HF_TOKEN_READ=hf_...                       # token with access to the gated models
    python3 model_generations.py                      # all languages
    python3 model_generations.py --languages ukrainian,german
    python3 model_generations.py --rerun              # regenerate results that already exist
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
DATA_DIR    = "global_eclektic_unfiltered"   # one CSV per language
FILE_SUFFIX = " - Questions.csv"             # "Ukrainian - Questions.csv" -> language "ukrainian"
RESULTS_DIR = "results"                      # final per-language results
WORK_DIR    = "work"                         # filtered datasets + per-model results
LOG_DIR     = "logs"                         # one log file per model

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


# ── Languages and paths ───────────────────────────────────────────────────────
def language_slug(name: str) -> str:
    """'Ukrainian' -> 'ukrainian', 'Brazilian Portuguese' -> 'brazilian_portuguese'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def discover_languages(data_dir: str = DATA_DIR) -> dict:
    """Map language slug -> CSV path for every CSV in data_dir."""
    languages = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        name = fname[: -len(FILE_SUFFIX)] if fname.endswith(FILE_SUFFIX) else os.path.splitext(fname)[0]
        slug = language_slug(name)
        if not slug:
            continue
        if slug in languages:
            raise SystemExit(f"[ERROR] Two files map to language '{slug}': "
                             f"{languages[slug]} and {os.path.join(data_dir, fname)}")
        languages[slug] = os.path.join(data_dir, fname)
    return languages


def safe_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def filtered_path(lang: str) -> str:
    return os.path.join(WORK_DIR, lang, "filtered_dataset.csv")


def model_result_path(model_id: str, lang: str) -> str:
    return os.path.join(WORK_DIR, lang, f"gen_{safe_name(model_id)}.csv")


def results_path(lang: str) -> str:
    return os.path.join(RESULTS_DIR, f"{lang}.csv")


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df)} raw rows.")
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
    print(f"  Kept {len(df)} of {n_start} rows.")
    return df


def read_filtered(lang: str) -> pd.DataFrame:
    return pd.read_csv(filtered_path(lang), encoding="utf-8", dtype=str, keep_default_na=False)


# ── Scoring ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize(prediction) == normalize(gold))


# ── Worker: one model, all its languages, in its own process on its own GPUs ─
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


def run_worker(model_id: str, n_gpus: int, langs: list) -> None:
    from vllm import LLM, SamplingParams

    name = safe_name(model_id)
    datasets = {lang: read_filtered(lang) for lang in langs}
    n_rows = sum(len(df) for df in datasets.values())
    print(f"[{name}] {len(langs)} languages, {n_rows} rows, "
          f"tensor_parallel_size={n_gpus}", flush=True)

    llm = LLM(
        model=model_id,
        tensor_parallel_size=n_gpus,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    # Every language x variant x question, generated in a single batched call
    prompts, keys = [], []
    for lang, df in datasets.items():
        for variant, (question_col, _) in VARIANTS.items():
            for q in df[question_col]:
                prompts.append({"prompt_token_ids": build_prompt_ids(tokenizer, q, model_id)})
                keys.append((lang, variant))

    start = time.time()
    outputs = llm.generate(prompts, params)
    print(f"[{name}] generated {len(outputs)} answers in {time.time() - start:.0f}s", flush=True)

    gens = {(lang, variant): [] for lang in datasets for variant in VARIANTS}
    for key, out in zip(keys, outputs):
        gens[key].append(out.outputs[0].text.strip())

    for lang, df in datasets.items():
        result = pd.DataFrame({"ID": df["ID"]})
        for variant, (_, answer_col) in VARIANTS.items():
            g = gens[(lang, variant)]
            result[f"gen_{name}_{variant}"] = g
            result[f"em_{name}_{variant}"] = [
                exact_match(p, gold) for p, gold in zip(g, df[answer_col])
            ]
            print(f"[{name}] {lang} Exact Match [{variant}]: "
                  f"{result[f'em_{name}_{variant}'].mean():.2%}", flush=True)

        out_path = model_result_path(model_id, lang)
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


def is_done(model_id: str, lang: str, df: pd.DataFrame) -> bool:
    """Done if this model's result file for this language exists and matches the dataset."""
    path = model_result_path(model_id, lang)
    if not os.path.exists(path):
        return False
    prev = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str}, keep_default_na=False)
    return prev["ID"].tolist() == df["ID"].tolist()


def print_failure(log_path: str) -> None:
    """Print the actual error lines (root cause) from a failed model's log."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    error_lines = []
    for line in lines:
        text = re.sub(r"^\([^)]*\)\s*", "", line.strip())  # drop "(EngineCore pid=…)" prefix
        if re.search(r"\b\w*(Error|Exception)\w*: ", text) and text not in error_lines:
            error_lines.append(text)
    shown = error_lines[-12:] if error_lines else lines[-15:]
    print("          " + "\n          ".join(shown))


def run_all_models(datasets: dict, rerun: bool) -> list:
    os.makedirs(LOG_DIR, exist_ok=True)
    total = detect_gpus()
    print(f"Detected {total} GPUs.\n")

    # Which languages each model still needs
    pending = {}
    for model_id in MODELS:
        if GPUS_PER_MODEL[model_id] > total:
            raise SystemExit(f"[ERROR] {model_id} needs {GPUS_PER_MODEL[model_id]} GPUs, "
                             f"only {total} available.")
        todo = [lang for lang, df in datasets.items()
                if rerun or not is_done(model_id, lang, df)]
        if todo:
            pending[model_id] = todo
        else:
            print(f"  [skip] {safe_name(model_id)} already has results for every language "
                  f"(use --rerun to regenerate)")

    free_gpus = list(range(total))
    running = {}   # model_id -> (Popen, gpu list, log file, start time)
    failed = []

    while pending or running:
        # Start every pending model that fits on the free GPUs, in MODELS order
        for model_id in list(pending):
            n = GPUS_PER_MODEL[model_id]
            if n <= len(free_gpus):
                gpus, free_gpus = free_gpus[:n], free_gpus[n:]
                langs = pending.pop(model_id)
                log_path = os.path.join(LOG_DIR, f"{safe_name(model_id)}.log")
                log = open(log_path, "w", encoding="utf-8")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
                proc = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__),
                     "--worker", model_id,
                     "--gpus", str(n),
                     "--langs", ",".join(langs)],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                running[model_id] = (proc, gpus, log, time.time())
                print(f"  [start] {safe_name(model_id)} on GPUs {gpus}, "
                      f"{len(langs)} languages  (log: {log_path})")

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
                print_failure(log_path)

        time.sleep(5)

    return failed


# ── Merge per-model results into one file per language ────────────────────────
def merge_results(lang: str, df: pd.DataFrame) -> pd.DataFrame:
    # Keep every dataset column except Exclude (already applied in filter_dataset)
    results = df.drop(columns="Exclude", errors="ignore").copy()

    for model_id in MODELS:
        if not is_done(model_id, lang, df):
            print(f"  [WARN] {lang}: no results for {safe_name(model_id)}, left out.")
            continue
        model_df = pd.read_csv(model_result_path(model_id, lang), encoding="utf-8-sig",
                               dtype={"ID": str}, keep_default_na=False)
        results = pd.concat([results, model_df.drop(columns="ID")], axis=1)

    em_cols = [c for c in results.columns if c.startswith("em_")]
    for col in em_cols:
        results[col] = results[col].astype(int)

    print(f"\n{lang}: Exact Match per model")
    for col in em_cols:
        print(f"  {col.replace('em_', '')}: {results[col].mean():.2%}")

    # Summary row
    summary = {"ID": "", "Question": "** MEAN EM **", "Answer": ""}
    for col in em_cols:
        summary[col] = round(results[col].mean(), 4)
    return pd.concat([results, pd.DataFrame([summary])], ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def prepare_datasets(languages: dict) -> dict:
    """Filter every language file and save it for the workers. Returns slug -> DataFrame."""
    datasets = {}
    for lang, path in languages.items():
        print(f"\n── {lang} " + "─" * max(0, 60 - len(lang)))
        try:
            df = filter_dataset(load_dataset(path))
        except (KeyError, pd.errors.ParserError, UnicodeDecodeError) as e:
            print(f"  [SKIP] {lang}: {e}")
            continue
        if df.empty:
            print(f"  [SKIP] {lang}: no rows left after filtering.")
            continue
        os.makedirs(os.path.dirname(filtered_path(lang)), exist_ok=True)
        df.to_csv(filtered_path(lang), index=False, encoding="utf-8")
        datasets[lang] = read_filtered(lang)   # read back so workers and merge see identical data
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="regenerate results that already exist")
    parser.add_argument("--languages",
                        help="comma-separated languages to run, e.g. ukrainian,german (default: all)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--langs", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.gpus, args.langs.split(","))
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

    languages = discover_languages()
    if args.languages:
        wanted = [language_slug(x) for x in args.languages.split(",") if x.strip()]
        unknown = [w for w in wanted if w not in languages]
        if unknown:
            raise SystemExit(f"[ERROR] Unknown languages {unknown}. Available: {list(languages)}")
        languages = {w: languages[w] for w in wanted}
    print(f"Found {len(languages)} languages in {DATA_DIR}: {', '.join(languages)}")

    datasets = prepare_datasets(languages)
    if not datasets:
        raise SystemExit("[ERROR] No usable language files.")
    print(f"\nRunning {len(MODELS)} models on {len(datasets)} languages.")

    failed = run_all_models(datasets, args.rerun)
    if failed:
        print(f"\n[WARN] Failed models: {[safe_name(m) for m in failed]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for lang, df in datasets.items():
        merge_results(lang, df).to_csv(results_path(lang), index=False, encoding="utf-8-sig")
        print(f"  saved {results_path(lang)}")

    print("\nDone. Upload with:  python3 upload.py")


if __name__ == "__main__":
    main()

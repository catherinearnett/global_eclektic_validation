"""
Global ECLeKTic pipeline: generate model answers, select the top 100 questions per
language, and upload to mrlbenchmarks/global_eclektic with one split per language.

Steps:
  generate  For every language CSV in global_eclektic_unfiltered/, run each model on the
            original question and its English translation. Scoring: both answers are
            lowercased with punctuation and extra spaces removed, and a generation is
            correct (em_* = 1) if it contains the expected answer as whole words.
            Uses vLLM; models run in parallel on separate GPUs.
            -> results/<language>.csv
  select    Per language, keep the top 100 questions by spread in the original-language
            scores across models. The translation pairs become "Question Translation" and
            "Answer Translation" (corrected if available, otherwise automatic).
            -> top100/<language>.csv
  upload    Push results/ and top100/ to mrlbenchmarks/global_eclektic as two subsets
            ("generations", the default, and "top100"), one split per language.

A language that is already uploaded is never generated, selected or uploaded again.
To redo one, delete its file from the repo first. If the repo can't be checked
(no token, no access, not created yet), the script stops rather than risk re-running.
A language whose results are missing any model is not selected or uploaded.

Setup (needs an NVIDIA driver supporting CUDA 12.x):
    uv venv eclektic --python 3.11 && source eclektic/bin/activate
    export VLLM_VERSION=0.30.0
    uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      --extra-index-url https://download.pytorch.org/whl/cu129
    uv pip install pandas huggingface_hub

Usage:
    export HF_TOKEN_READ=hf_...       # model downloads (gated models) and reading the repo
    export HF_TOKEN_UPLOAD=hf_...     # uploading
    python3 global_eclektic.py                          # all steps: generate, select, upload
    python3 global_eclektic.py generate                 # one step: generate | select | upload
    python3 global_eclektic.py --languages ukrainian,german
    python3 global_eclektic.py generate --rerun         # regenerate local results not yet uploaded
    python3 global_eclektic.py select --source hf       # select from generations already on HF
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time
import unicodedata

os.environ["PYTHONIOENCODING"] = "utf-8"

# Hugging Face libraries (vLLM, transformers) only read HF_TOKEN, so copy the
# read token into it. Needed for gated models such as Llama.
if os.environ.get("HF_TOKEN_READ"):
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN_READ"]

import pandas as pd

# ══ Config ════════════════════════════════════════════════════════════════════
DATA_DIR    = "global_eclektic_unfiltered"   # one CSV per language
FILE_SUFFIX = " - Questions.csv"             # "Ukrainian - Questions.csv" -> language "ukrainian"
WORK_DIR    = "work"                         # filtered datasets + per-model results
LOG_DIR     = "logs"                         # one log file per model

# Hugging Face dataset repo
REPO_ID = "mrlbenchmarks/global_eclektic"
# subset name -> (local folder, folder in the repo)
SUBSETS = {
    "generations": ("results", "data/generations"),
    "top100":      ("top100",  "data/top100"),
}
DEFAULT_SUBSET = "generations"
# Tokens tried in this order to read the repo (it may be private)
READ_TOKEN_VARS = ("HF_TOKEN_UPLOAD", "HF_TOKEN_READ", "HF_TOKEN_MRL_READ")

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

# Dataset columns to keep, in this order ("Exclude" is used for filtering, then dropped)
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
OUTPUT_COLUMNS = [c for c in KEEP_COLUMNS if c != "Exclude"]

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

# Selection
TOP_N = 100
SAMPLE_VARIANT = "orig"    # which variant drives the selection: "orig" or "en"

SUMMARY_LABEL = "** MEAN EM **"

# In the top100 files, each translation pair becomes one column (same order as
# TRANSLATION_PAIRS): the corrected translation, or the automatic one if there's none
TOP100_TRANSLATION_COLUMNS = ["Question Translation", "Answer Translation"]

README_BODY = """# Global ECLeKTic

Each language is its own split. Subsets:

- `generations`: every filtered question with each model's answers (`gen_*`) and
  exact-match scores (`em_*`), for the original question (`_orig`) and its English
  translation (`_en`).
- `top100`: per language, the 100 questions with the largest spread in
  original-language scores across models.
"""


# ══ Shared helpers ════════════════════════════════════════════════════════════
def section(title: str) -> None:
    print(f"\n══ {title} " + "═" * max(0, 66 - len(title)))


def language_slug(name: str) -> str:
    """'Ukrainian' -> 'ukrainian', 'Brazilian Portuguese' -> 'brazilian_portuguese'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def safe_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def wanted_languages(args) -> list:
    """Languages named with --languages, as slugs (empty = all)."""
    if not args.languages:
        return []
    return [language_slug(x) for x in args.languages.split(",") if x.strip()]


def missing_models(df: pd.DataFrame) -> list:
    """Models without both em_<model>_<variant> columns in a results table."""
    return [
        safe_name(m) for m in MODELS
        if any(f"em_{safe_name(m)}_{v}" not in df.columns for v in VARIANTS)
    ]


# ── Hugging Face repo checks ──────────────────────────────────────────────────
def read_token() -> str:
    for var in READ_TOKEN_VARS:
        if os.environ.get(var):
            return os.environ[var]
    raise SystemExit(
        f"[ERROR] No Hugging Face token set ({', '.join(READ_TOKEN_VARS)}). "
        f"One is needed to check {REPO_ID} for languages that were already uploaded."
    )


def repo_files() -> set:
    """All files in the repo. Stops the script if the repo can't be checked, so that
    already-uploaded languages are never re-run by mistake."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        return set(HfApi(token=read_token()).list_repo_files(REPO_ID, repo_type="dataset"))
    except RepositoryNotFoundError:
        raise SystemExit(
            f"[ERROR] {REPO_ID} was not found, or the token can't see it.\n"
            f"        If it doesn't exist yet, create it first:\n"
            f"          hf repo create {REPO_ID} --repo-type dataset --private"
        )
    except Exception as e:
        raise SystemExit(
            f"[ERROR] Could not check {REPO_ID} for uploaded languages: {e}\n"
            f"        Stopping so that already-uploaded languages aren't re-run."
        )


def uploaded_languages(subset: str, files: set) -> set:
    """Languages that already have a file for this subset in the repo."""
    repo_dir = SUBSETS[subset][1]
    return {
        os.path.splitext(os.path.basename(f))[0]
        for f in files
        if f.startswith(repo_dir + "/") and f.endswith(".csv")
    }


# ══ Step 1: generate ══════════════════════════════════════════════════════════
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


def filtered_path(lang: str) -> str:
    return os.path.join(WORK_DIR, lang, "filtered_dataset.csv")


def model_result_path(model_id: str, lang: str) -> str:
    return os.path.join(WORK_DIR, lang, f"gen_{safe_name(model_id)}.csv")


def results_path(lang: str) -> str:
    return os.path.join(SUBSETS["generations"][0], f"{lang}.csv")


def top_path(lang: str) -> str:
    return os.path.join(SUBSETS["top100"][0], f"{lang}.csv")


# ── Load and filter one language ──────────────────────────────────────────────
def load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df)} raw rows.")
    return df


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


# ── Scoring ───────────────────────────────────────────────────────────────────
def normalize(text) -> str:
    """Lowercase, turn punctuation into spaces, and collapse whitespace.
    'Jean-Paul Sartre.' -> 'jean paul sartre'"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ʼ", " ")          # Ukrainian apostrophe ʼ counts as a letter otherwise
    text = re.sub(r"[^\w\s]|_", " ", text)      # punctuation -> space
    return " ".join(text.split())


def flexible_match(prediction, gold) -> int:
    """1 if the normalised gold answer appears in the normalised generation as whole
    words ('kyiv' matches 'the capital is kyiv', not 'kyivska'), else 0."""
    gold_norm = normalize(gold)
    if not gold_norm:
        return 0
    return int(f" {gold_norm} " in f" {normalize(prediction)} ")


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
                flexible_match(p, gold) for p, gold in zip(g, df[answer_col])
            ]
            print(f"[{name}] {lang} accuracy [{variant}]: "
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
    print(f"\nDetected {total} GPUs.")

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
            print(f"  [skip] {safe_name(model_id)} already has local results for every language "
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


def merge_results(lang: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dataset columns + every model's gen/em columns + a summary row."""
    results = df[OUTPUT_COLUMNS].copy()

    for model_id in MODELS:
        if not is_done(model_id, lang, df):
            print(f"  [WARN] {lang}: no results for {safe_name(model_id)}, left out.")
            continue
        model_df = pd.read_csv(model_result_path(model_id, lang), encoding="utf-8-sig",
                               dtype={"ID": str}, keep_default_na=False)
        # Re-score from the saved generations, so a scoring change applies
        # to existing local results without regenerating
        name = safe_name(model_id)
        for variant, (_, answer_col) in VARIANTS.items():
            model_df[f"em_{name}_{variant}"] = [
                flexible_match(p, gold)
                for p, gold in zip(model_df[f"gen_{name}_{variant}"], df[answer_col])
            ]
        results = pd.concat([results, model_df.drop(columns="ID")], axis=1)

    em_cols = [c for c in results.columns if c.startswith("em_")]
    for col in em_cols:
        results[col] = results[col].astype(int)

    print(f"\n{lang}: accuracy per model")
    for col in em_cols:
        print(f"  {col.replace('em_', '')}: {results[col].mean():.2%}")

    summary = {"ID": "", "Question": SUMMARY_LABEL, "Answer": ""}
    for col in em_cols:
        summary[col] = round(results[col].mean(), 4)
    return pd.concat([results, pd.DataFrame([summary])], ignore_index=True)


def step_generate(args, files: set) -> None:
    section("generate")

    # Fail fast if vLLM isn't installed in this Python (workers use the same one)
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            f"[ERROR] vLLM is not installed for {sys.executable}\n"
            f"        Install it with:  {sys.executable} -m pip install -U vllm"
        )
    if not os.environ.get("HF_TOKEN"):
        print("[WARN] HF_TOKEN_READ is not set; gated models (e.g. Llama) will fail to download.")

    languages = discover_languages()
    wanted = wanted_languages(args)
    if wanted:
        unknown = [w for w in wanted if w not in languages]
        if unknown:
            raise SystemExit(f"[ERROR] Unknown languages {unknown}. Available: {list(languages)}")
        languages = {w: languages[w] for w in wanted}
    print(f"Found {len(languages)} languages in {DATA_DIR}: {', '.join(languages)}")

    # Never re-run a language whose generations are already uploaded
    uploaded = uploaded_languages("generations", files)
    done = [lang for lang in languages if lang in uploaded]
    if done:
        print(f"  [skip] Already uploaded to {REPO_ID}, not re-run: {', '.join(done)}")
        languages = {lang: path for lang, path in languages.items() if lang not in uploaded}
    if not languages:
        print(f"Nothing to generate: every language is already uploaded.")
        return

    datasets = prepare_datasets(languages)
    if not datasets:
        print("Nothing to generate: no usable language files.")
        return
    print(f"\nRunning {len(MODELS)} models on {len(datasets)} languages.")

    failed = run_all_models(datasets, args.rerun)
    if failed:
        print(f"\n[WARN] Failed models: {[safe_name(m) for m in failed]}")

    os.makedirs(SUBSETS["generations"][0], exist_ok=True)
    for lang, df in datasets.items():
        merge_results(lang, df).to_csv(results_path(lang), index=False, encoding="utf-8-sig")
        print(f"  saved {results_path(lang)}")


# ══ Step 2: select ════════════════════════════════════════════════════════════
def load_results(lang: str, source: str) -> pd.DataFrame:
    path = results_path(lang)
    if source == "hf":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO_ID, f"{SUBSETS['generations'][1]}/{lang}.csv",
                               repo_type="dataset", token=read_token())
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str})
    return df[df["Question"] != SUMMARY_LABEL].reset_index(drop=True)


def select_top(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    print(f"\n── {lang} " + "─" * max(0, 60 - len(lang)))
    print(f"  Loaded {len(df)} questions.")

    missing = missing_models(df)
    if missing:
        print(f"  [SKIP] Results are missing models {missing}; regenerate before selecting.")
        return None

    dataset_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    gen_cols     = [c for c in df.columns if c.startswith("gen_")]
    em_cols_all  = [c for c in df.columns if c.startswith("em_")]
    em_cols      = [c for c in em_cols_all if c.endswith(f"_{SAMPLE_VARIANT}")]

    df[em_cols_all] = (
        df[em_cols_all].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )

    # Count correct models per variant (saved for both, selection uses one)
    for variant in VARIANTS:
        variant_cols = [c for c in em_cols_all if c.endswith(f"_{variant}")]
        df[f"total_correct_{variant}"] = df[variant_cols].sum(axis=1)

    # Remove questions all models got right (selection variant only)
    df_filtered = df[df[f"total_correct_{SAMPLE_VARIANT}"] < len(em_cols)].copy()
    print(f"  After removing all-correct: {len(df_filtered)} questions remaining.")

    # Spread = variance across model scores (maximised when some get it right, some don't)
    df_filtered["spread"] = df_filtered[em_cols].var(axis=1)
    df_filtered["any_correct"] = df_filtered[em_cols].max(axis=1)

    # Sort: prioritise spread (discriminative), then questions at least one got right
    df_filtered = df_filtered.sort_values(by=["spread", "any_correct"], ascending=[False, False])

    top = df_filtered.head(TOP_N).reset_index(drop=True)
    print(f"  Selected top {len(top)} questions by spread.")
    if len(top) < TOP_N:
        print(f"  [INFO] Fewer than {TOP_N} questions available for {lang}.")

    if len(top):
        print(f"  Spread: max {top['spread'].max():.4f}, min {top['spread'].min():.4f}, "
              f"mean {top['spread'].mean():.4f}; ≥1 correct: {top['any_correct'].sum()}")
        print(f"  Per-model accuracy on top {len(top)}:")
        for col in em_cols_all:
            tag = " (used for selection)" if col in em_cols else ""
            print(f"    {col.replace('em_', '')}: {top[col].mean():.2%}{tag}")

    out_cols = dataset_cols + gen_cols + em_cols_all + [
        f"total_correct_{v}" for v in VARIANTS
    ] + ["spread"]
    return merge_translations(top[out_cols])


def merge_translations(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each automatic/corrected translation pair with a single column:
    the corrected translation if there is one, otherwise the automatic one."""
    df = df.copy()
    for (auto_col, corr_col), new_col in zip(TRANSLATION_PAIRS, TOP100_TRANSLATION_COLUMNS):
        corrected = df[corr_col] if corr_col in df else pd.Series(pd.NA, index=df.index)
        automatic = df[auto_col] if auto_col in df else pd.Series(pd.NA, index=df.index)
        merged = corrected.where(~_is_blank(corrected), automatic)
        present = [c for c in (auto_col, corr_col) if c in df]
        position = min(df.columns.get_loc(c) for c in present) if present else len(df.columns)
        df = df.drop(columns=present)
        df.insert(position, new_col, merged)
    return df


def step_select(args, files: set) -> None:
    section("select")

    if args.source == "local":
        local_dir = SUBSETS["generations"][0]
        langs = sorted(os.path.splitext(f)[0] for f in os.listdir(local_dir)
                       if f.endswith(".csv")) if os.path.isdir(local_dir) else []
    else:
        langs = sorted(uploaded_languages("generations", files))
    wanted = wanted_languages(args)
    if wanted:
        langs = [lang for lang in langs if lang in wanted]
    if not langs:
        print(f"Nothing to select: no results found ({args.source}).")
        return
    print(f"Found results for {len(langs)} languages ({args.source}): {', '.join(langs)}")

    # Never redo a language whose top 100 is already uploaded
    uploaded = uploaded_languages("top100", files)
    done = [lang for lang in langs if lang in uploaded]
    if done:
        print(f"  [skip] Top {TOP_N} already uploaded to {REPO_ID}, not redone: {', '.join(done)}")
        langs = [lang for lang in langs if lang not in uploaded]

    os.makedirs(SUBSETS["top100"][0], exist_ok=True)
    for lang in langs:
        top = select_top(load_results(lang, args.source), lang)
        if top is not None:
            top.to_csv(top_path(lang), index=False, encoding="utf-8-sig")
            print(f"  Saved to {top_path(lang)}")


# ══ Step 3: upload ════════════════════════════════════════════════════════════
def build_configs(files: set) -> list:
    """One config per subset, one split per language file found in the repo."""
    configs = []
    for subset, (_, repo_dir) in SUBSETS.items():
        paths = sorted(f for f in files if f.startswith(repo_dir + "/") and f.endswith(".csv"))
        if not paths:
            continue
        config = {"config_name": subset}
        if subset == DEFAULT_SUBSET:
            config["default"] = True
        config["data_files"] = [
            {"split": os.path.splitext(os.path.basename(p))[0], "path": p} for p in paths
        ]
        configs.append(config)
    return configs


def build_readme(existing: str, configs: list) -> str:
    """Replace the `configs` entry in the README's YAML header, keeping everything else."""
    import yaml

    meta, body = {}, README_BODY
    if existing is not None:
        if existing.startswith("---"):
            _, front, rest = existing.split("---", 2)
            meta = yaml.safe_load(front) or {}
            body = rest.lstrip("\n")
        else:
            body = existing
    meta["configs"] = configs
    header = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return f"---\n{header}---\n\n{body}"


def step_upload(args, files: set) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    section("upload")
    token = os.environ.get("HF_TOKEN_UPLOAD")
    if not token:
        raise SystemExit("[ERROR] HF_TOKEN_UPLOAD environment variable not set.")
    wanted = wanted_languages(args)

    # Collect local files that aren't in the repo yet; never overwrite an uploaded language
    operations = []
    for subset, (local_dir, repo_dir) in SUBSETS.items():
        if not os.path.isdir(local_dir):
            print(f"  [skip] {subset}: no {local_dir}/ folder")
            continue
        uploaded = uploaded_languages(subset, files)
        new, skipped, incomplete = [], [], []
        for fname in sorted(f for f in os.listdir(local_dir) if f.endswith(".csv")):
            lang = os.path.splitext(fname)[0]
            if wanted and lang not in wanted:
                continue
            if lang in uploaded:
                skipped.append(lang)
                continue
            path = os.path.join(local_dir, fname)
            if missing_models(pd.read_csv(path, encoding="utf-8-sig", nrows=1)):
                incomplete.append(lang)
                continue
            new.append(lang)
            operations.append(CommitOperationAdd(path_in_repo=f"{repo_dir}/{fname}",
                                                 path_or_fileobj=path))
        if skipped:
            print(f"  [skip] {subset}: already uploaded, not overwritten: {', '.join(skipped)}")
        if incomplete:
            print(f"  [skip] {subset}: missing some models' results, not uploaded: "
                  f"{', '.join(incomplete)}")
        print(f"  {subset}: uploading {len(new)} languages" + (f" ({', '.join(new)})" if new else ""))
    if not operations:
        print("Nothing new to upload.")
        return

    # README lists every language file in the repo, including ones uploaded earlier
    existing_readme = None
    if "README.md" in files:
        path = hf_hub_download(REPO_ID, "README.md", repo_type="dataset", token=token)
        with open(path, encoding="utf-8") as f:
            existing_readme = f.read()
    all_files = files | {op.path_in_repo for op in operations}
    readme = build_readme(existing_readme, build_configs(all_files))
    operations.append(CommitOperationAdd(path_in_repo="README.md",
                                         path_or_fileobj=readme.encode("utf-8")))

    HfApi(token=token).create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message="Upload per-language results",
    )
    print(f"Uploaded {len(operations) - 1} files to hf.co/datasets/{REPO_ID}")


# ══ Main ══════════════════════════════════════════════════════════════════════
STEPS = {"generate": step_generate, "select": step_select, "upload": step_upload}


def main():
    parser = argparse.ArgumentParser(
        description="Generate, select and upload Global ECLeKTic results.")
    parser.add_argument("step", nargs="?", default="all", choices=["all", *STEPS],
                        help="step to run (default: all, in order)")
    parser.add_argument("--languages",
                        help="comma-separated languages, e.g. ukrainian,german (default: all)")
    parser.add_argument("--rerun", action="store_true",
                        help="generate: regenerate local results that aren't uploaded yet")
    parser.add_argument("--source", choices=["local", "hf"], default="local",
                        help="select: read results from results/ (local) or the HF repo (hf)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--langs", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.gpus, args.langs.split(","))
        return

    steps = list(STEPS) if args.step == "all" else [args.step]
    for step in steps:
        # Check the repo fresh before each step; stops here if it can't be checked
        STEPS[step](args, repo_files())


if __name__ == "__main__":
    main()

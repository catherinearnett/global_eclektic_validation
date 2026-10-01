"""
Accuracy on the top 100 questions, per language and model: original-language
question vs. English translation.

Reads the "top100" subset (one file per language) from the HF repo by default,
or from the local top100/ folder written by global_eclektic.py.

Note: the top 100 were selected using the original-language scores (questions
every model got right in the original language were removed), so original-language
accuracy here is lower by construction. Compare the two columns with that in mind.

Output:
    prints a table per language
    top100_accuracy.csv with one row per language and model, plus averages:
      model "ALL MODELS"        = mean over models within a language
      language "ALL LANGUAGES"  = mean over languages (each language weighted equally)

Usage:
    export HF_TOKEN_UPLOAD=hf_...              # any token that can read the repo
                                               # (HF_TOKEN_UPLOAD, HF_TOKEN_READ or HF_TOKEN_MRL_READ)
    python3 top100_accuracy.py                 # from the HF repo
    python3 top100_accuracy.py --source local  # from top100/
    python3 top100_accuracy.py --languages ukr_cyrl,ita_latn
"""

import argparse
import os
import re

import pandas as pd

REPO_ID   = "mrlbenchmarks/global_eclektic"
HF_DIR    = "data/top100"
LOCAL_DIR = "top100"
OUT_PATH  = "top100_accuracy.csv"

VARIANTS = {"orig": "original", "en": "english"}
READ_TOKEN_VARS = ("HF_TOKEN_UPLOAD", "HF_TOKEN_READ", "HF_TOKEN_MRL_READ")
_token = None   # the first token that can read the repo, found by list_languages


def read_token() -> str:
    if _token:
        return _token
    raise SystemExit(f"[ERROR] No Hugging Face token set ({', '.join(READ_TOKEN_VARS)}).")


def list_languages(source: str) -> dict:
    """Map language -> local path or repo filename."""
    global _token
    if source == "local":
        if not os.path.isdir(LOCAL_DIR):
            raise SystemExit(f"[ERROR] {LOCAL_DIR}/ not found.")
        return {os.path.splitext(f)[0]: os.path.join(LOCAL_DIR, f)
                for f in sorted(os.listdir(LOCAL_DIR)) if f.endswith(".csv")}

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    tokens = [(var, os.environ[var]) for var in READ_TOKEN_VARS if os.environ.get(var)]
    if not tokens:
        raise SystemExit(f"[ERROR] No Hugging Face token set ({', '.join(READ_TOKEN_VARS)}).")
    for var, token in tokens:   # use the first token that can see the repo
        try:
            files = HfApi(token=token).list_repo_files(REPO_ID, repo_type="dataset")
        except RepositoryNotFoundError:
            continue
        _token = token
        print(f"Reading {REPO_ID} with {var}.")
        return {os.path.splitext(os.path.basename(f))[0]: f
                for f in sorted(files) if f.startswith(HF_DIR + "/") and f.endswith(".csv")}
    raise SystemExit(
        f"[ERROR] {REPO_ID} was not found with any of: {', '.join(v for v, _ in tokens)}.\n"
        f"        Either nothing has been uploaded yet (run: python3 global_eclektic.py upload),\n"
        f"        or none of these tokens has access to the repo.\n"
        f"        To use local results instead: python3 top100_accuracy.py --source local"
    )


def load(location: str, source: str) -> pd.DataFrame:
    if source == "hf":
        from huggingface_hub import hf_hub_download
        location = hf_hub_download(REPO_ID, location, repo_type="dataset", token=read_token())
    return pd.read_csv(location, encoding="utf-8-sig", dtype={"ID": str})


def language_accuracy(df: pd.DataFrame, lang: str) -> list:
    """One row per model with original and English accuracy."""
    pattern = re.compile(r"^em_(.+)_(" + "|".join(VARIANTS) + r")$")
    models = sorted({m.group(1) for c in df.columns if (m := pattern.match(c))})
    rows = []
    for model in models:
        row = {"language": lang, "model": model, "n_questions": len(df)}
        for variant, label in VARIANTS.items():
            col = f"em_{model}_{variant}"
            scores = pd.to_numeric(df[col], errors="coerce") if col in df else None
            row[f"acc_{label}"] = scores.mean() if scores is not None else float("nan")
        row["diff_english_minus_original"] = row["acc_english"] - row["acc_original"]
        rows.append(row)
    if rows:
        avg = {"language": lang, "model": "ALL MODELS", "n_questions": len(df)}
        for key in ("acc_original", "acc_english", "diff_english_minus_original"):
            avg[key] = pd.Series([r[key] for r in rows]).mean()
        rows.append(avg)
    return rows


def print_table(rows: list, title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    print(f"  {'model':<32} {'n':>4} {'original':>9} {'english':>9} {'diff':>8}")
    for r in rows:
        print(f"  {r['model']:<32} {r['n_questions']:>4} {r['acc_original']:>9.2%} "
              f"{r['acc_english']:>9.2%} {r['diff_english_minus_original']:>+8.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--source", choices=["hf", "local"], default="hf",
                        help="read the top100 files from the HF repo (default) or top100/")
    parser.add_argument("--languages", help="comma-separated languages (default: all)")
    args = parser.parse_args()

    languages = list_languages(args.source)
    if args.languages:
        wanted = [re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")
                  for x in args.languages.split(",") if x.strip()]
        missing = [w for w in wanted if w not in languages]
        if missing:
            print(f"[WARN] No top100 file for: {', '.join(missing)}")
        languages = {k: v for k, v in languages.items() if k in wanted}
    if not languages:
        raise SystemExit(f"[ERROR] No top100 files found ({args.source}).")
    print(f"Top 100 accuracy for {len(languages)} languages ({args.source}): {', '.join(languages)}")

    rows = []
    for lang, location in languages.items():
        lang_rows = language_accuracy(load(location, args.source), lang)
        if not lang_rows:
            print(f"\n[SKIP] {lang}: no em_ columns found.")
            continue
        print_table(lang_rows, lang)
        rows.extend(lang_rows)

    # Mean over languages, each language weighted equally
    table = pd.DataFrame(rows)
    overall = []
    for model, group in table.groupby("model", sort=False):
        overall.append({
            "language": "ALL LANGUAGES",
            "model": model,
            "n_questions": int(group["n_questions"].sum()),
            "acc_original": group["acc_original"].mean(),
            "acc_english": group["acc_english"].mean(),
            "diff_english_minus_original": group["diff_english_minus_original"].mean(),
        })
    if len(languages) > 1:
        print_table(overall, f"ALL LANGUAGES (mean of {table['language'].nunique()} languages)")

    table = pd.concat([table, pd.DataFrame(overall)], ignore_index=True)
    table.round(4).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

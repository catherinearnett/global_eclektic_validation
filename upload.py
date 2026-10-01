"""
Upload per-language results to the HuggingFace dataset repo, one split per language.

Repo layout:
    data/generations/<language>.csv   from results/<language>.csv   (subset "generations", the default)
    data/top100/<language>.csv        from top100/<language>.csv    (subset "top100")

The README's YAML header is rewritten so every language file in the repo is listed
as its own split. Languages uploaded earlier stay listed, and any other README
content and metadata are kept.

Load it with:
    load_dataset("mrlbenchmarks/global_eclektic", split="ukrainian")             # generations
    load_dataset("mrlbenchmarks/global_eclektic", "top100", split="ukrainian")

Usage:
    export HF_TOKEN_UPLOAD=hf_...
    python3 upload.py
"""

import os

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO_ID = "mrlbenchmarks/global_eclektic"

# subset name -> (local folder, folder in the repo)
SUBSETS = {
    "generations": ("results", "data/generations"),
    "top100":      ("top100",  "data/top100"),
}
DEFAULT_SUBSET = "generations"

DEFAULT_README_BODY = """# Global ECLeKTic

Each language is its own split. Subsets:

- `generations`: every filtered question with each model's answers (`gen_*`) and
  exact-match scores (`em_*`), for the original question (`_orig`) and its English
  translation (`_en`).
- `top100`: per language, the 100 questions with the largest spread in
  original-language scores across models.
"""


def build_configs(repo_files: set) -> list:
    """One config per subset, one split per language file found in the repo."""
    configs = []
    for subset, (_, repo_dir) in SUBSETS.items():
        files = sorted(f for f in repo_files if f.startswith(repo_dir + "/") and f.endswith(".csv"))
        if not files:
            continue
        config = {"config_name": subset}
        if subset == DEFAULT_SUBSET:
            config["default"] = True
        config["data_files"] = [
            {"split": os.path.splitext(os.path.basename(f))[0], "path": f} for f in files
        ]
        configs.append(config)
    return configs


def build_readme(existing: str, configs: list) -> str:
    """Replace the `configs` entry in the README's YAML header, keeping everything else."""
    meta, body = {}, DEFAULT_README_BODY
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


def main():
    token = os.environ.get("HF_TOKEN_UPLOAD")
    if not token:
        raise SystemExit("[ERROR] HF_TOKEN_UPLOAD environment variable not set.")

    # Collect local files to upload
    operations = []
    for subset, (local_dir, repo_dir) in SUBSETS.items():
        if not os.path.isdir(local_dir):
            print(f"  [skip] {subset}: no {local_dir}/ folder")
            continue
        files = sorted(f for f in os.listdir(local_dir) if f.endswith(".csv"))
        for fname in files:
            operations.append(CommitOperationAdd(
                path_in_repo=f"{repo_dir}/{fname}",
                path_or_fileobj=os.path.join(local_dir, fname),
            ))
        print(f"  {subset}: {len(files)} languages from {local_dir}/ "
              f"({', '.join(os.path.splitext(f)[0] for f in files)})")
    if not operations:
        raise SystemExit("[ERROR] Nothing to upload; run model_generations.py first.")

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    # README lists every language file in the repo, including ones uploaded earlier
    repo_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    existing_readme = None
    if "README.md" in repo_files:
        path = hf_hub_download(REPO_ID, "README.md", repo_type="dataset", token=token)
        with open(path, encoding="utf-8") as f:
            existing_readme = f.read()
    all_files = repo_files | {op.path_in_repo for op in operations}
    readme = build_readme(existing_readme, build_configs(all_files))
    operations.append(CommitOperationAdd(path_in_repo="README.md",
                                         path_or_fileobj=readme.encode("utf-8")))

    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message="Upload per-language results",
    )
    print(f"Uploaded {len(operations) - 1} files to hf.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

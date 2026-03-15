#!/usr/bin/env python3
"""
upload_dataset_to_hf.py
────────────────────────
Merges all JSONL files in a directory and pushes them to HuggingFace
as a dataset repository. Run this once locally before using the Colab notebook.

Usage:
  pip install datasets huggingface_hub
  python upload_dataset_to_hf.py --dir ~/dataset --repo YOUR_USERNAME/robot-navigation-dataset

The dataset is pushed as private by default. Change --public to make it public.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter

from datasets import Dataset, Features, Value
from huggingface_hub import login, HfApi


def load_all_jsonl(directory: Path):
    samples = []
    jsonl_files = sorted(directory.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {directory}")

    for path in jsonl_files:
        count_before = len(samples)
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] {path.name} line {i+1} skipped: {e}")
        print(f"  {path.name}: {len(samples) - count_before} samples")

    return samples


def validate_samples(samples):
    required = {'image_b64', 'lidar_b64', 'intent'}
    valid_intents = {'FORWARD', 'LEFT', 'RIGHT', 'REVERSE', 'STOP'}
    issues = 0
    for i, s in enumerate(samples):
        missing = required - s.keys()
        if missing:
            print(f"  [WARN] Sample {i} missing fields: {missing}")
            issues += 1
        if s.get('intent') not in valid_intents:
            print(f"  [WARN] Sample {i} invalid intent: {s.get('intent')}")
            issues += 1
    if issues:
        print(f"\n  {issues} issues found — fix before uploading.")
        raise ValueError("Dataset validation failed.")
    print(f"  All {len(samples)} samples valid.")


def main():
    parser = argparse.ArgumentParser(description="Upload robot dataset to HuggingFace Hub.")
    parser.add_argument("--dir",    required=True, help="Directory containing .jsonl files")
    parser.add_argument("--repo",   required=True, help="HF repo name, e.g. username/robot-navigation-dataset")
    parser.add_argument("--public", action="store_true", help="Make the dataset public (default: private)")
    parser.add_argument("--token",  default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    dataset_dir = Path(args.dir).expanduser()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {dataset_dir}")

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = args.token or os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        login()   # interactive prompt

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading JSONL files from {dataset_dir} ...")
    samples = load_all_jsonl(dataset_dir)
    print(f"Total samples loaded: {len(samples)}")

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\nValidating...")
    validate_samples(samples)

    # ── Class distribution ────────────────────────────────────────────────────
    counter = Counter(s['intent'] for s in samples)
    print("\nClass distribution:")
    for intent, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = count / len(samples) * 100
        bar = '█' * int(pct / 2)
        print(f"  {intent:<10} {count:>5}  {pct:>5.1f}%  {bar}")

    # ── Build HF Dataset ──────────────────────────────────────────────────────
    print("\nBuilding HuggingFace dataset...")
    features = Features({
        "image_b64": Value("string"),
        "lidar_b64": Value("string"),
        "intent":    Value("string"),
    })

    hf_dataset = Dataset.from_list(
        [{"image_b64": s["image_b64"],
          "lidar_b64": s["lidar_b64"],
          "intent":    s["intent"]}
         for s in samples],
        features=features,
    )

    total_mb = sum(
        len(s["image_b64"]) + len(s["lidar_b64"])
        for s in samples
    ) / (1024 ** 2) * 0.75   # base64 → bytes ~= *0.75

    print(f"Dataset size: {len(hf_dataset)} rows, ~{total_mb:.0f} MB raw images")

    # ── Push ──────────────────────────────────────────────────────────────────
    private = not args.public
    print(f"\nPushing to {args.repo} ({'private' if private else 'public'}) ...")
    hf_dataset.push_to_hub(
        args.repo,
        private=private,
        commit_message=f"Upload {len(hf_dataset)} robot navigation samples",
    )

    print(f"\nDone!")
    print(f"Dataset URL: https://huggingface.co/datasets/{args.repo}")
    print(f"\nIn your Colab notebook set:")
    print(f'  HF_DATASET_REPO = "{args.repo}"')


if __name__ == "__main__":
    main()
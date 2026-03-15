#!/usr/bin/env python3
"""
inspect_dataset.py
──────────────────
Sanity-check a collected .jsonl dataset (camera + LiDAR format).

Usage:
  python3 inspect_dataset.py path/to/dataset_YYYYMMDD_HHMMSS.jsonl

Outputs:
  - Total sample count and file size
  - Per-intent class distribution with ASCII bar chart
  - Class imbalance warning if ratio > 5x
  - Preview grid saved as <dataset>_preview.jpg
    Each cell shows camera image (left) + LiDAR render (right) with intent label
"""

import sys
import json
import base64
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


INTENTS = ['FORWARD', 'REVERSE', 'LEFT', 'RIGHT', 'STOP']


def load_jsonl(path: Path):
    samples = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'  [WARN] Line {i+1} skipped: {e}')
    return samples


def decode_b64_image(b64: str) -> np.ndarray:
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def print_distribution(counter: Counter, total: int):
    bar_width = 30
    print(f'\n{"Intent":<10} {"Count":>6}  {"Pct":>6}  Bar')
    print('─' * 58)
    for intent in INTENTS:
        count = counter.get(intent, 0)
        pct   = count / total * 100 if total else 0
        bar   = '█' * int(pct / 100 * bar_width)
        print(f'{intent:<10} {count:>6}  {pct:>5.1f}%  {bar}')
    for k, v in counter.items():
        if k not in INTENTS:
            print(f'{k:<10} {v:>6}  ← unexpected label')
    print()


def make_preview(samples, out_path: Path, grid=(3, 3)):
    """
    Each grid cell = camera image (left half) + LiDAR render (right half),
    with the intent label overlaid. Makes it easy to visually verify
    that the correct label was recorded for each scene.
    """
    rows, cols = grid
    n = rows * cols
    idxs = [int(i * len(samples) / n) for i in range(n)]

    thumb_w, thumb_h = 320, 120   # per cell: 160px camera + 160px lidar
    half_w = thumb_w // 2

    cells = []
    for idx in idxs:
        s = samples[idx]

        # Camera
        cam = decode_b64_image(s.get('image_b64', ''))
        if cam is None:
            cam = np.zeros((thumb_h, half_w, 3), dtype=np.uint8)
        else:
            cam = cv2.resize(cam, (half_w, thumb_h))

        # LiDAR
        lid = decode_b64_image(s.get('lidar_b64', ''))
        if lid is None:
            lid = np.zeros((thumb_h, half_w, 3), dtype=np.uint8)
        else:
            lid = cv2.resize(lid, (half_w, thumb_h))

        cell = np.hstack([cam, lid])

        # Intent label
        intent = s.get('intent', '?')
        color  = {
            'FORWARD': (0, 220, 0),
            'LEFT':    (220, 200, 0),
            'RIGHT':   (0, 200, 220),
            'REVERSE': (0, 80, 220),
            'STOP':    (0, 0, 220),
        }.get(intent, (200, 200, 200))
        cv2.putText(cell, intent, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Separator line between cam and lidar
        cv2.line(cell, (half_w, 0), (half_w, thumb_h), (80, 80, 80), 1)

        cells.append(cell)

    row_imgs = [np.hstack(cells[i * cols:(i + 1) * cols]) for i in range(rows)]
    grid_img = np.vstack(row_imgs)
    cv2.imwrite(str(out_path), grid_img)
    print(f'Preview saved → {out_path}')
    print('  (left half of each cell = camera, right half = LiDAR render)')


def main():
    parser = argparse.ArgumentParser(
        description='Inspect a teleop dataset .jsonl file or directory (camera + LiDAR).')
    parser.add_argument('jsonl', help='Path to .jsonl dataset file or directory')
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f'Path not found: {path}')
        sys.exit(1)

    jsonl_files = []
    if path.is_dir():
        jsonl_files = sorted(list(path.glob('*.jsonl')))
        if not jsonl_files:
            print(f'No .jsonl files found in directory: {path}')
            sys.exit(1)
    else:
        jsonl_files = [path]

    samples = []
    total_size = 0
    for f_path in jsonl_files:
        total_size += f_path.stat().st_size
        print(f'Loading : {f_path}')
        samples.extend(load_jsonl(f_path))

    size_mb = total_size / (1024 ** 2)
    print(f'\nTotal Size    : {size_mb:.1f} MB')

    total = len(samples)
    print(f'Total Samples : {total}')

    if total == 0:
        print('No samples found. Exiting.')
        sys.exit(0)

    # Check which fields are present
    first = samples[0]
    has_lidar = 'lidar_b64' in first
    print(f'Fields  : image_b64=yes  lidar_b64={"yes" if has_lidar else "NO -- monocular dataset"}')

    counter = Counter(s.get('intent', 'UNKNOWN') for s in samples)
    print_distribution(counter, total)

    # Class balance check
    present = {k: v for k, v in counter.items() if k in INTENTS}
    if present:
        min_cls = min(present.values())
        max_cls = max(present.values())
        ratio   = max_cls / min_cls if min_cls else float('inf')
        if ratio > 5:
            min_intent = min(present, key=present.get)
            max_intent = max(present, key=present.get)
            print(f'WARNING  High class imbalance: {max_intent}({max_cls}) vs '
                  f'{min_intent}({min_cls}) = {ratio:.1f}x ratio.')
            print('  Tip: drive more laps focusing on turns and stops,')
            print('       or use weighted sampling during fine-tuning.\n')
        else:
            print(f'OK  Class balance ratio: {ratio:.1f}x — looks good.\n')

    # Preview grid
    if path.is_dir():
        preview_path = path / 'dataset_preview.jpg'
    else:
        preview_path = path.parent / (path.stem + '_preview.jpg')

    try:
        make_preview(samples, preview_path)
    except Exception as e:
        print(f'Preview generation failed: {e}')


if __name__ == '__main__':
    main()

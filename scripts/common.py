"""Shared data loading and measurements for cosine HNSW experiments."""
import csv
import json
from pathlib import Path

import numpy as np


def load_matrix(path, kind="vectors"):
    path = Path(path)
    if path.suffix == ".npy":
        a = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix in (".fvecs", ".ivecs"):
        raw = np.memmap(path, dtype="<i4", mode="r")
        d = int(raw[0]) if len(raw) else 0
        if d <= 0 or len(raw) % (d + 1):
            raise ValueError(f"Invalid vector file: {path}")
        rows = raw.reshape(-1, d + 1)
        if not np.all(rows[:, 0] == d):
            raise ValueError(f"Inconsistent dimensions: {path}")
        a = rows[:, 1:]
        if path.suffix == ".fvecs":
            a = a.view("<f4")
    else:
        raise ValueError("Use .npy, .fvecs, or .ivecs")
    if a.ndim != 2 or not all(a.shape):
        raise ValueError("Expected a nonempty 2D matrix")
    if kind == "vectors":
        if not np.issubdtype(a.dtype, np.floating):
            raise ValueError("Vectors must have a floating-point dtype")
        return np.asarray(a, dtype=np.float32, order="C")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Ground truth must contain integer row IDs")
    return a


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recall_at_k(labels, truth, k):
    # Unique IDs ensure repeated predictions cannot inflate recall.
    return sum(len(set(a) & set(b)) for a, b in zip(labels, truth[:, :k])) / (len(labels) * k)


def adc_at_recall(curve, target):
    # Each graph is interpolated separately. No extrapolation past measured recall.
    points = {}
    for row in curve:
        recall, adc = row["recall"], row["adc"]
        points[recall] = min(adc, points.get(recall, float("inf")))
    xs = sorted(points)
    if target < xs[0] or target > xs[-1]:
        return None
    return float(np.interp(target, xs, [points[x] for x in xs]))

#!/usr/bin/env python3
"""Measure recall and ADC on one saved index using the common query reader."""
import argparse
from pathlib import Path
import sys

import numpy as np

from common import dump, load_matrix, read, recall_at_k, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp"))
import ada_hnswlib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--groundtruth", type=Path, required=True)
    p.add_argument("--efs", type=int, nargs="+", default=[10, 20, 40, 80, 120, 200, 400, 800])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, default=24)
    a = p.parse_args()
    if a.k < 1 or a.threads < 1 or min(a.efs) < a.k:
        p.error("Require k>=1, threads>=1, and every efSearch>=k")
    build = read(a.index_dir / "build.json")
    q, gt = load_matrix(a.query), load_matrix(a.groundtruth, "ids")
    if q.shape[1] != build["dim"] or len(gt) != len(q) or gt.shape[1] < a.k or a.k > build["n"]:
        p.error("Query, ground truth, or k does not match the index")
    if np.any(gt[:, :a.k] < 0) or np.any(gt[:, :a.k] >= build["n"]):
        p.error("Ground truth contains IDs outside the index")
    reader = ada_hnswlib.Index("cosine", q.shape[1])
    reader.load_index(str(a.index_dir / "graph.index"))
    rows = []
    for ef in sorted(set(a.efs)):
        reader.set_ef(ef)
        labels, _, calls = reader.knn_query(q, k=a.k, num_threads=a.threads,
                                           return_distance_computations=True)
        rows.append(dict(method=build["method"], M=build["M"], E0=build["E0"],
            seed=build["seed"], ef_search=ef, k=a.k, recall=recall_at_k(labels, gt, a.k),
            adc=calls / len(q), distance_computations=int(calls)))
    dump(a.index_dir / "query_config.json", dict(query=str(a.query.resolve()),
        groundtruth=str(a.groundtruth.resolve()), k=a.k, threads=a.threads,
        efs=sorted(set(a.efs)), query_rows=len(q)))
    dump(a.index_dir / "curves.json", rows)
    write_csv(a.index_dir / "curves.csv", rows)


if __name__ == "__main__":
    main()

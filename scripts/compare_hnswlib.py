#!/usr/bin/env python3
"""Run a matched-seed grid comparing b-only construction with upstream hnswlib."""
import argparse
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np

from common import adc_at_recall, dump, read, write_csv

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--groundtruth", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    p.add_argument("--M", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--efc", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    p.add_argument("--efs", type=int, nargs="+", default=[10, 20, 40, 80, 120, 200, 400, 800])
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--keep-indexes", action="store_true", help="retain every graph after querying")
    a = p.parse_args()
    if (a.threads < 1 or a.k < 1 or min(a.M) < 2 or max(a.M) > 10000 or
            min(a.efc) < max(a.M) or max(a.efc) > 1000000 or
            min(a.efs) < a.k or min(a.seeds) < 0):
        p.error("Invalid grid, seed, thread count, or query parameters")
    for values in (a.M, a.efc, a.seeds, a.efs):
        if len(values) != len(set(values)):
            p.error("Grid values and seeds must be unique")
    for path in (a.base, a.query, a.groundtruth):
        if not path.is_file():
            p.error(f"Missing input: {path}")
    a.output.mkdir(parents=True, exist_ok=False)
    dump(a.output / "config.json", {k: str(v) if isinstance(v, Path) else v
                                   for k, v in vars(a).items()})
    (a.output / "STATUS").write_text("RUNNING\n")
    per_seed, curves = [], []
    try:
        for m, efc in product(a.M, a.efc):
            for position, seed in enumerate(a.seeds):
                methods = ["hnswlib", "b_only"] if position % 2 == 0 else ["b_only", "hnswlib"]
                for method in methods:
                    out = a.output / f"M{m}_E{efc}_s{seed}" / method
                    subprocess.run([sys.executable, str(HERE / "build_index.py"),
                        "--method", method, "--base", str(a.base), "--output", str(out),
                        "--M", str(m), "--efc", str(efc), "--seed", str(seed),
                        "--threads", str(a.threads)], check=True)
                    subprocess.run([sys.executable, str(HERE / "query_index.py"),
                        "--index-dir", str(out), "--query", str(a.query),
                        "--groundtruth", str(a.groundtruth), "--threads", str(a.threads),
                        "--k", str(a.k), "--efs", *map(str, a.efs)], check=True)
                    build, curve = read(out / "build.json"), read(out / "curves.json")
                    per_seed.append(dict(method=method, M=m, E0=efc, seed=seed,
                        build_seconds=build["build_seconds"],
                        ADC95=adc_at_recall(curve, .95), ADC98=adc_at_recall(curve, .98)))
                    curves.extend(curve)
                    write_csv(a.output / "per_seed.csv", per_seed)
                    write_csv(a.output / "recall_adc.csv", curves)
                    if not a.keep_indexes:
                        (out / "graph.index").unlink()
        summary = []
        for m, efc, method in product(a.M, a.efc, ["hnswlib", "b_only"]):
            rows = [r for r in per_seed if (r["M"], r["E0"], r["method"]) == (m, efc, method)]
            result = dict(method=method, M=m, E0=efc, seeds=len(rows))
            times = [r["build_seconds"] for r in rows]
            result.update(build_seconds_mean=float(np.mean(times)),
                          build_seconds_std=float(np.std(times, ddof=1)) if len(times) > 1 else 0.)
            for metric in ["ADC95", "ADC98"]:
                values = [r[metric] for r in rows]
                result[metric] = float(np.mean(values)) if all(v is not None for v in values) else None
            summary.append(result)
        write_csv(a.output / "summary.csv", summary)
        dump(a.output / "summary.json", summary)
        (a.output / "STATUS").write_text("COMPLETE\n")
    except BaseException:
        (a.output / "STATUS").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()

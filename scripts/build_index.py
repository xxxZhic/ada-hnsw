#!/usr/bin/env python3
"""Build one b-only or upstream hnswlib cosine index."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import time

from common import dump, load_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpp"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["b_only", "hnswlib"], default="b_only")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--efc", type=int, default=200)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--threads", type=int, default=24)
    a = p.parse_args()
    if a.M < 2 or a.M > 10000 or a.efc < a.M or a.efc > 1000000 or a.threads < 1 or a.seed < 0:
        p.error("Require 2<=M<=10000, M<=efc<=1000000, threads>=1, seed>=0")
    if a.method == "b_only":
        import ada_hnswlib as engine
        version = engine.__version__
    else:
        import hnswlib as engine
        version = importlib.metadata.version("hnswlib")
    a.output.mkdir(parents=True, exist_ok=False)
    dump(a.output / "config.json", dict(method=a.method, M=a.M, E0=a.efc,
        seed=a.seed, threads=a.threads, base=str(a.base.resolve()), distance="cosine",
        host=platform.node(), platform=platform.platform(), engine_version=version,
        extension_sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in sorted((ROOT / "cpp").rglob("*.h"))
                      if "build" not in f.relative_to(ROOT / "cpp").parts}))
    (a.output / "STATUS").write_text("RUNNING\n")
    try:
        start = time.perf_counter()
        x = load_matrix(a.base)
        index = engine.Index("cosine", x.shape[1])
        index.init_index(len(x), M=a.M, ef_construction=a.efc, random_seed=a.seed)
        if a.method == "b_only":
            result = dict(index.build_b_only(x, num_threads=a.threads))
        else:
            index.add_items(x, num_threads=a.threads)
            result = {}
        seconds = time.perf_counter() - start
        if index.get_current_count() != len(x):
            raise RuntimeError("Incomplete index")
        index.save_index(str(a.output / "graph.index"))
        result.update(method=a.method, n=len(x), dim=x.shape[1], M=a.M, E0=a.efc,
            seed=a.seed, threads=a.threads, build_seconds=seconds,
            timing_scope="input loading, conversion, index initialization and construction, including normalization; excludes serialization",
            index_bytes=(a.output / "graph.index").stat().st_size)
        dump(a.output / "build.json", result)
        (a.output / "STATUS").write_text("COMPLETE\n")
        print(f"{a.method}: {seconds:.3f} s, {len(x)} vectors", flush=True)
    except BaseException:
        (a.output / "STATUS").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()

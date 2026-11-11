"""Exercise input formats, interpolation, and a complete paired grid."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import adc_at_recall, load_matrix, read


class ScriptTests(unittest.TestCase):
    def test_interpolation_does_not_extrapolate(self):
        curve = [dict(recall=.90, adc=100), dict(recall=.98, adc=300)]
        self.assertAlmostEqual(adc_at_recall(curve, .95), 225)
        self.assertIsNone(adc_at_recall(curve, .99))

    def test_input_formats(self):
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            encoded = np.column_stack([np.full(6, 4, dtype=np.int32), x.view(np.int32)])
            encoded.astype("<i4").tofile(p / "x.fvecs")
            np.testing.assert_array_equal(load_matrix(p / "x.fvecs"), x)
            encoded[1, 0] = 3
            encoded.tofile(p / "invalid.fvecs")
            with self.assertRaises(ValueError):
                load_matrix(p / "invalid.fvecs")

    def test_paired_grid(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=(1200, 16)).astype(np.float32)
        q = rng.normal(size=(40, 16)).astype(np.float32)
        xn = x / np.linalg.norm(x, axis=1, keepdims=True)
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        gt = np.argsort(-(qn @ xn.T), axis=1)[:, :10].astype(np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name, data in [("base", x), ("query", q), ("gt", gt)]:
                np.save(p / f"{name}.npy", data)
            subprocess.run([sys.executable, str(ROOT / "scripts/compare_hnswlib.py"),
                "--base", str(p / "base.npy"), "--query", str(p / "query.npy"),
                "--groundtruth", str(p / "gt.npy"), "--output", str(p / "run"),
                "--M", "8", "--efc", "40", "80", "--seeds", "43", "44",
                "--efs", "10", "40", "80", "--threads", "1"], check=True)
            with (p / "run/per_seed.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["method"] for r in rows}, {"hnswlib", "b_only"})
            self.assertEqual(len(read(p / "run/summary.json")), 4)
            self.assertEqual((p / "run/STATUS").read_text(), "COMPLETE\n")
            self.assertFalse(list((p / "run").rglob("*.index")))


if __name__ == "__main__":
    unittest.main()

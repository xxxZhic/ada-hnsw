"""Adaptive construction, parameter validation, and stock-reader compatibility."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import hnswlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpp"))
import ada_hnswlib


class NativeTests(unittest.TestCase):
    def test_default_blocks_and_stock_reader(self):
        # Cross the fixed prefix and two adaptive blocks using the real defaults.
        x = np.random.default_rng(70).normal(size=(70000, 24)).astype(np.float32)
        q = np.random.default_rng(71).normal(size=(100, 24)).astype(np.float32)
        index = ada_hnswlib.Index("cosine", 24)
        index.init_index(len(x), M=12, ef_construction=80, random_seed=43)
        levels = index.preview_levels(len(x))
        np.testing.assert_array_equal(levels, index.preview_levels(len(x)))
        result = index.build_b_only(x, num_threads=2, level_values=levels)
        histogram = {int(k): v for k, v in result["ef_histogram"].items()}
        self.assertEqual(result["method"], "b_only")
        self.assertEqual(sum(histogram.values()), len(x))
        self.assertEqual(sum(k*v for k, v in histogram.items()), result["ef_sum"])
        self.assertEqual(result["fixed_prefix_count"], 33268)
        self.assertEqual(result["allocation_count"], len(x) - 33268)
        self.assertEqual((result["ef_low"], result["ef_high"]), (40, 92))
        self.assertGreater(len(histogram), 10)
        self.assertGreaterEqual(histogram.get(80, 0), 33268)
        self.assertGreater(result["distance_computation_sum"], len(x))
        self.assertEqual(result["upper_ef"], 80)
        blocks = result["blocks"]
        self.assertEqual([b["fixed_reference"] for b in blocks], [True, True, False, False])
        self.assertGreater(blocks[2]["normalized_std_used"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "graph.index")
            index.save_index(path)
            stock = hnswlib.Index("cosine", 24)
            stock.load_index(path)
            loaded = ada_hnswlib.Index("cosine", 24)
            loaded.load_index(path)
            for ef in [10, 40, 100]:
                loaded.set_ef(ef)
                stock.set_ef(ef)
                labels, distances, calls = loaded.knn_query(q, k=10, num_threads=2,
                    return_distance_computations=True)
                expected, expected_distances = stock.knn_query(q, k=10, num_threads=2)
                np.testing.assert_array_equal(labels, expected)
                np.testing.assert_allclose(distances, expected_distances, atol=1e-6, rtol=1e-6)
                _, _, single_calls = loaded.knn_query(q, k=10, num_threads=1,
                    return_distance_computations=True)
                self.assertEqual(calls, single_calls)

    def test_stock_built_index_reader(self):
        x = np.random.default_rng(1).normal(size=(1000, 16)).astype(np.float32)
        stock = hnswlib.Index("cosine", 16)
        stock.init_index(len(x), M=8, ef_construction=40, random_seed=43)
        stock.add_items(x, num_threads=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "stock.index")
            stock.save_index(path)
            reader = ada_hnswlib.Index("cosine", 16)
            reader.load_index(path)
            stock.set_ef(40)
            reader.set_ef(40)
            a, ad = reader.knn_query(x[:20], k=10, num_threads=1)
            b, bd = stock.knn_query(x[:20], k=10, num_threads=1)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(ad, bd, atol=1e-6)

    def test_frozen_calibration_and_custom_blocks(self):
        x = np.random.default_rng(2).normal(size=(1600, 16)).astype(np.float32)
        index = ada_hnswlib.Index("cosine", 16)
        index.init_index(len(x), M=8, ef_construction=40)
        r = index.build_b_only(x, num_threads=2, n_ref=32, block_size=256, update_weight=0)
        self.assertEqual(r["fixed_prefix_count"], 288)
        active = [b for b in r["blocks"] if not b["fixed_reference"]]
        self.assertGreater(len(active), 2)
        self.assertEqual(len({b["normalized_mean_used"] for b in active}), 1)
        self.assertEqual(len({b["normalized_std_used"] for b in active}), 1)

    def test_small_index_and_invalid_inputs(self):
        x = np.ones((32, 8), dtype=np.float32)
        for method in ["build_b_only", "build_fixed"]:
            index = ada_hnswlib.Index("cosine", 8)
            index.init_index(len(x), M=4, ef_construction=20)
            self.assertEqual(getattr(index, method)(x, num_threads=1)["ef_histogram"], {"20": 32})
            with self.assertRaises(ValueError):
                getattr(index, method)(x)
        for options in [dict(block_size=0), dict(n_ref=0), dict(high_factor=.9),
                        dict(low_factor=-1), dict(update_weight=1.1), dict(z_clip=0),
                        dict(checkpoint_factor=0), dict(update_weight=float("nan"))]:
            index = ada_hnswlib.Index("cosine", 8)
            index.init_index(len(x), M=4, ef_construction=20)
            with self.assertRaises(ValueError):
                index.build_b_only(x, **options)
            self.assertEqual(index.get_current_count(), 0)
        with self.assertRaises(ValueError):
            ada_hnswlib.Index("l2", 8)


if __name__ == "__main__":
    unittest.main()

# Ada-Construction

Adaptive HNSW index construction for cosine distance. This release contains
the **b-only method** and scripts for comparison with upstream **hnswlib**.

## Method

For each incoming vector, the level-0 search starts with capacity `U`. After
`T` node expansions, we observe the distance `delta` to the closest candidate
awaiting expansion. We normalize it using the incoming vector's estimated
distance mean `m` and standard deviation `sigma`:

```text
h = (delta - m) / sigma
z = (h - historical_mean_h) / historical_std_h
```

The score `z` selects the current result-set capacity for the rest of that
insertion. Larger scores receive larger capacities. Search continues with
the existing candidate queue and visited state. `M` remains fixed.

Coordinate means and variances are estimated from the first 500 vectors.
These vectors and the following reference block are inserted with the
initial `efConstruction`, denoted by `E0`. The reference block initializes
the mean and variance of `h`. Subsequent blocks update these statistics
after all their insertions finish. Worker threads share the same calibration
within each block.

Default parameters:

| Parameter | Default |
|---|---:|
| Minimum capacity `L` | `max(M, round(0.50 * E0))` |
| Maximum capacity `U` | `max(E0, round(1.15 * E0))` |
| Checkpoint `T` | `max(1, floor(0.50 * E0 + 0.5))` |
| Coordinate reference size | 500 |
| Calibration block size | 32,768 |
| Calibration update weight | 0.75 |
| Score clipping range | `[-2.5, 2.5]` |

The clipped score maps linearly to `[L, U]`. Capacities are rounded to the
nearest integer, with ties to even. Upper-layer connection searches use `E0`.

## Installation

Use Linux, Python 3.10 or newer, and a C++17 compiler with OpenMP support.

```bash
python -m pip install -r requirements.txt
(cd cpp && python setup.py build_ext --inplace)
python -m unittest discover -s tests -v
```

The native extension is `ada_hnswlib`. Set `HNSWLIB_NO_NATIVE=1` when building
an extension that must run on CPUs with different instruction sets.
The baseline dependency is upstream `hnswlib==0.8.0`.

## Input data

Pass the base vectors, queries, and exact cosine ground truth explicitly:

- Base and query vectors: float32 `.npy` matrices or `.fvecs` files.
- Ground truth: integer `.npy` matrices or `.ivecs` files. Each row contains
  the nearest base-vector IDs for one query, ordered by distance.
- IDs are zero-based row indices in the base matrix.

Vectors are normalized during construction and querying. Ground truth must
use cosine distance on the same base collection and contain at least `k`
neighbors per query. Data files are supplied separately.

## Compare with hnswlib

Run one configuration using three matched construction seeds:

```bash
python scripts/compare_hnswlib.py \
  --base /path/to/base.npy \
  --query /path/to/query.npy \
  --groundtruth /path/to/cosine_gt.ivecs \
  --output results/comparison \
  --M 16 --efc 200 --seeds 43 44 45 \
  --efs 10 20 40 80 120 200 400 800 \
  --threads 24
```

For the main construction grid, replace the parameter line with:

```text
--M 8 12 16 --efc 100 200 400 --seeds 43 44 45
```

The script runs both methods on the same machine with matching inputs,
`M`, `E0`, seeds, and thread counts. It alternates build order across seeds.
Each build runs in a separate process. Construction time and query metrics
come from the same built index. Threads are scheduled by the operating system.
Parallel construction can produce different graphs even with the same seed.

Construction time includes input loading and conversion, index initialization,
normalization, and construction. For our method it also includes the coordinate
statistics and calibration updates. Serialization is excluded.

Both saved indexes are queried using the same hnswlib-compatible reader with
distance counting. Tests check its results against upstream hnswlib. The
scripts report Recall@10 and ADC, the average distance computations per query.
Use `--k` to change the recall cutoff.

Outputs:

| File | Contents |
|---|---|
| `per_seed.csv` | Construction time, ADC95, and ADC98 for each configuration and seed |
| `recall_adc.csv` | Measured recall and ADC at every `efSearch` |
| `summary.csv`, `summary.json` | Mean construction time and its sample standard deviation, mean ADC95 and ADC98 |
| Each method's directory | Build statistics, configuration, and query results |

ADC95 and ADC98 are interpolated on each graph's measured curve at recall
0.95 and 0.98, then averaged across seeds. An empty entry means the target
falls outside at least one graph's measured recall range. All measured
configurations and seeds are retained in the output tables.

Use a new output directory for each run. By default, the script removes each
newly built index after recording its query results to limit disk usage.
Add `--keep-indexes` to retain the graphs.

## Build or query an individual index

```bash
python scripts/build_index.py --method b_only \
  --base /path/to/base.npy --output results/b_only \
  --M 16 --efc 200 --seed 43 --threads 24

python scripts/query_index.py --index-dir results/b_only \
  --query /path/to/query.npy --groundtruth /path/to/cosine_gt.ivecs \
  --efs 10 20 40 80 120 200 400 800 --threads 24
```

Use `--method hnswlib` and a new output directory to build the baseline.

## Python interface

```python
import sys
sys.path.insert(0, "cpp")
import ada_hnswlib

index = ada_hnswlib.Index("cosine", base.shape[1])
index.init_index(len(base), M=16, ef_construction=200, random_seed=43)
statistics = index.build_b_only(base, num_threads=24)
index.set_ef(200)
labels, distances = index.knn_query(queries, k=10, num_threads=24)
index.save_index("graph.index")
```

`build_b_only` also accepts `low_factor`, `high_factor`, `checkpoint_factor`,
`n_ref`, `block_size`, `update_weight`, and `z_clip`. Their defaults are listed
above. Saved indexes use the standard hnswlib index layout.

The modified hnswlib code and its license are identified in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

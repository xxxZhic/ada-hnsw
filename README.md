# Ada-Construction

Adaptive HNSW index construction for cosine distance. Our method adjusts
`efConstruction` during each insertion using observations from the ongoing
search, while keeping `M` fixed.

## Installation

Requires Linux, Python 3.10 or newer, and a C++17 compiler with OpenMP support.

```bash
git clone https://github.com/xxxZhic/ada-hnsw.git
cd ada-hnsw
python -m pip install -r requirements.txt
(cd cpp && python setup.py build_ext --inplace)
```

## Data

- Base and query vectors: float32 `.npy` matrices or `.fvecs` files.
- Ground truth: integer `.npy` matrices or `.ivecs` files containing the
  exact nearest-neighbor IDs for each query under cosine distance.
- IDs are zero-based row indices in the base matrix. Each ground-truth row
  must contain at least 10 neighbors, ordered by distance.

Vectors are normalized automatically during construction and querying.

## Compare with hnswlib

Run both methods with the same inputs, parameters, seeds, and thread count:

```bash
python scripts/compare_hnswlib.py \
  --base /path/to/base.npy \
  --query /path/to/query.npy \
  --groundtruth /path/to/groundtruth.ivecs \
  --output results/comparison \
  --M 16 --efc 200 --seeds 43 44 45 \
  --efs 10 20 40 80 120 200 400 800 \
  --threads 24
```

For the full construction grid, replace the `--M` and `--efc` arguments with:

```text
--M 8 12 16 --efc 100 200 400
```

Results are saved in the output directory:

- `summary.csv`: mean construction time, ADC95, and ADC98 across seeds.
- `per_seed.csv`: results for each configuration and seed.
- `recall_adc.csv`: Recall@10 and ADC at each query setting.

ADC is the average number of distance computations per query.
ADC95 and ADC98 measure this cost at Recall@10 of 0.95 and 0.98.
Timing covers data loading, normalization, and index construction.
Use a new output directory for each run. Add `--keep-indexes` to retain
the indexes after querying.

## Tests

```bash
python -m unittest discover -s tests -v
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for hnswlib attribution
and licensing.

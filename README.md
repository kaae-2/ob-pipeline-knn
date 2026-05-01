# KNN Module

## What this module does

Provides a Python KNN-style baseline (prototype-based approximation) with
Omnibenchmark-compatible inputs and outputs.

This is an approximate prototype-mode runner, not an exact all-cells KNN.
Interpret its benchmark scores as a lightweight baseline with deliberate memory
guardrails.

- CLI: `knn_cli.py`
- Model logic: `knn_model.py`
- Local runner: `run_knn.sh`
- Output: `knn_predicted_labels.tar.gz`

## Run locally

```bash
bash models/knn/run_knn.sh
```

Safe tuning knobs for large folds:

- `KNN_N_JOBS` defaults to `4` in the main benchmark launcher.
- `KNN_PROTOTYPES_PER_CLASS` defaults to `8`.
- `KNN_TARGET_INDEX_BYTES_PER_BATCH` defaults to `1073741824` (1 GiB target for
  the temporary nearest-index array).
- `KNN_PREDICT_BATCH_SIZE` can cap row batches directly when a fold still pushes
  memory or timeout limits.

## Run as part of benchmark

Configured in `benchmark/Clustering_conda.yml` analysis stage and executed by:

```bash
just benchmark
```

## What `run_knn.sh` needs

- Preprocessing outputs at `models/knn/out/data/data_preprocessing/default`
- Python with `numpy`, `pandas`, `scikit-learn`
- Writable output directory `models/knn/out/data/analysis/default/knn`

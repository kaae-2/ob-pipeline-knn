# KNN Module

## What this module does

Provides a Python KNN-style baseline (prototype-based approximation) with
Omnibenchmark-compatible inputs and outputs.

- CLI: `knn_cli.py`
- Model logic: `knn_model.py`
- Local runner: `run_knn.sh`
- Output: `knn_predicted_labels.tar.gz`

## Run locally

```bash
bash models/knn/run_knn.sh
```

## Run as part of benchmark

Configured in `benchmark/Clustering_conda.yml` analysis stage and executed by:

```bash
just benchmark
```

## What `run_knn.sh` needs

- Preprocessing outputs at `models/knn/out/data/data_preprocessing/default`
- Python with `numpy`, `pandas`, `scikit-learn`
- Writable output directory `models/knn/out/data/analysis/default/knn`

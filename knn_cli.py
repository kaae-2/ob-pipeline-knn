#!/usr/bin/env python3
"""KNN model runner following the repository model I/O contract."""

from __future__ import annotations

import argparse
import gzip
import io
import math
import os
import re
import sys
import tarfile
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


UNLABELED_TOKENS = {"", "unlabeled", "ungated"}
PREDICT_BATCH_SIZE = 20_000
DEFAULT_RESERVED_CORES = 2


def _resolve_n_jobs() -> int:
    env_value = os.getenv("KNN_N_JOBS")
    if env_value is not None and env_value.strip() != "":
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - DEFAULT_RESERVED_CORES)


def _extract_sample_number(sample_name: str) -> Optional[str]:
    base = os.path.basename(sample_name)
    while True:
        root, ext = os.path.splitext(base)
        if not ext:
            break
        base = root
    match = re.search(r"(\d+)(?!.*\d)", base)
    if match:
        return match.group(1)
    return None


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_numeric = df.apply(pd.to_numeric, axis=0, errors="coerce")
    if df_numeric.empty:
        raise ValueError("Data matrix is empty.")
    df_numeric = df_numeric.dropna(axis=1, how="all")
    if df_numeric.empty:
        raise ValueError("Data matrix contains no numeric columns.")
    return df_numeric


def _read_csv_from_member(member_name: str, data: bytes) -> pd.DataFrame:
    if member_name.endswith(".gz"):
        data = gzip.decompress(data)
    df = pd.read_csv(io.BytesIO(data), header=None)
    df = _coerce_numeric(df)
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    return df


def _load_single_matrix(path: str) -> pd.DataFrame:
    if not tarfile.is_tarfile(path):
        df = pd.read_csv(path, header=None, compression="infer")
        df = _coerce_numeric(df)
        df.columns = [f"f{i}" for i in range(df.shape[1])]
        return df

    with tarfile.open(path, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
        if not members:
            raise ValueError(f"No files found in archive: {path}")
        file_obj = tar.extractfile(members[0])
        if file_obj is None:
            raise ValueError(f"Unable to read {members[0].name} from {path}")
        return _read_csv_from_member(members[0].name, file_obj.read())


def _normalize_labels(series: pd.Series) -> np.ndarray:
    raw = series.astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")

    has_non_numeric = numeric.isna() & ~raw.isna() & ~raw.eq("")
    if has_non_numeric.any():
        lowered = raw.str.lower()
        valid_mask = ~lowered.isin(UNLABELED_TOKENS)
        valid_labels = sorted(raw[valid_mask].dropna().unique())
        mapping = {label: idx + 1 for idx, label in enumerate(valid_labels)}
        mapped = raw.map(mapping).astype(float)
        mapped[~valid_mask] = float("nan")
        return mapped.to_numpy()

    labels = numeric.astype(float)
    labels = labels.mask(labels == 0)
    return labels.to_numpy()


def _load_labels(path: str) -> np.ndarray:
    if not tarfile.is_tarfile(path):
        series = pd.read_csv(
            path,
            header=None,
            compression="infer",
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        ).iloc[:, 0]
        return _normalize_labels(series)

    with tarfile.open(path, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
        if not members:
            raise ValueError(f"No files found in archive: {path}")
        file_obj = tar.extractfile(members[0])
        if file_obj is None:
            raise ValueError(f"Unable to read {members[0].name} from {path}")
        data = file_obj.read()
        if members[0].name.endswith(".gz"):
            data = gzip.decompress(data)
        series = pd.read_csv(
            io.BytesIO(data),
            header=None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        ).iloc[:, 0]
        return _normalize_labels(series)


def _load_test_samples(path: str) -> list[tuple[str, pd.DataFrame, Optional[str]]]:
    if not tarfile.is_tarfile(path):
        sample_name = os.path.basename(path)
        df = pd.read_csv(path, header=None, compression="infer")
        df = _coerce_numeric(df)
        df.columns = [f"f{i}" for i in range(df.shape[1])]
        return [(sample_name, df, _extract_sample_number(sample_name))]

    samples: list[tuple[str, pd.DataFrame, Optional[str]]] = []
    with tarfile.open(path, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
        for member in members:
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            df = _read_csv_from_member(member.name, file_obj.read())
            samples.append((member.name, df, _extract_sample_number(member.name)))

    if not samples:
        raise ValueError(f"No sample files found in archive: {path}")
    return samples


def _compute_k(total_cells: int, smallest_population_size: int, labeled_cells: int) -> int:
    if total_cells <= 0:
        raise ValueError("Training matrix has no rows.")
    if smallest_population_size <= 0:
        raise ValueError("Smallest labeled population size must be > 0.")
    raw_k = math.floor(total_cells / smallest_population_size)
    bounded_k = max(1, raw_k)
    return min(bounded_k, labeled_cells)


def _fit_model(
    train_matrix: pd.DataFrame, train_labels: np.ndarray, n_jobs: int
) -> tuple[KNeighborsClassifier, int]:
    if len(train_matrix) != len(train_labels):
        raise ValueError(
            "Number of labels does not match rows in training matrix "
            f"({len(train_labels)} != {len(train_matrix)})."
        )

    labels_series = pd.Series(train_labels)
    labeled_mask = labels_series.notna()
    labeled_labels = labels_series[labeled_mask].astype(int)
    if labeled_labels.empty:
        raise ValueError("No labeled rows available after excluding unlabeled class 0.")

    population_sizes = labeled_labels.value_counts()
    smallest_population_size = int(population_sizes.min())
    labeled_cells = int(labeled_mask.sum())
    total_cells = int(len(train_matrix))
    k = _compute_k(total_cells, smallest_population_size, labeled_cells)

    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
    classifier.fit(train_matrix.loc[labeled_mask].to_numpy(), labeled_labels.to_numpy())
    return classifier, k


def _predict_in_batches(
    model: KNeighborsClassifier, sample_matrix: np.ndarray, batch_size: int = PREDICT_BATCH_SIZE
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("Predict batch size must be > 0.")

    total_rows = sample_matrix.shape[0]
    if total_rows == 0:
        return np.array([], dtype=int)

    chunks: list[np.ndarray] = []
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        chunks.append(model.predict(sample_matrix[start_idx:end_idx]))
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="clustbench KNN runner")
    parser.add_argument("--data.train_matrix", type=str, required=True)
    parser.add_argument("--data.train_labels", type=str, required=True)
    parser.add_argument("--data.test_matrix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--name", type=str, default="clustbench")

    args = parser.parse_args()

    name = args.name
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    train_matrix = _load_single_matrix(getattr(args, "data.train_matrix"))
    train_labels = _load_labels(getattr(args, "data.train_labels"))
    test_samples = _load_test_samples(getattr(args, "data.test_matrix"))

    n_jobs = _resolve_n_jobs()

    model, k = _fit_model(train_matrix, train_labels, n_jobs=n_jobs)
    print(f"KNN: using computed k={k} with n_jobs={n_jobs}", flush=True)

    output_tar = os.path.join(output_dir, f"{name}_predicted_labels.tar.gz")
    if os.path.islink(output_tar):
        os.unlink(output_tar)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_files: list[str] = []
        for idx, (_, sample_df, sample_number) in enumerate(test_samples, start=1):
            predictions = _predict_in_batches(model, sample_df.to_numpy())
            out_labels = [str(int(label)) for label in predictions]

            if sample_number is None:
                sample_number = str(idx)
            file_name = f"{name}-prediction-{sample_number}.csv"
            file_path = os.path.join(tmpdir, file_name)
            pd.Series(out_labels).to_csv(file_path, index=False, header=False)
            output_files.append(file_path)

        with tarfile.open(output_tar, "w:gz") as tar:
            for path in output_files:
                tar.add(path, arcname=os.path.basename(path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)

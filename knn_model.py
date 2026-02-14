#!/usr/bin/env python3
"""Approximate KNN model training and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin


@dataclass(frozen=True)
class PrototypeModel:
    centers: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class FitStats:
    k: int
    n_jobs_effective: int
    prototype_count: int


def compute_k(
    total_cells: int,
    smallest_population_size: int,
    labeled_cells: int,
    num_labels: int,
) -> int:
    """Compute k with dataset- and class-based caps."""
    if total_cells <= 0:
        raise ValueError("Training matrix has no rows.")
    if smallest_population_size <= 0:
        raise ValueError("Smallest labeled population size must be > 0.")
    if num_labels <= 0:
        raise ValueError("Number of labeled classes must be > 0.")

    raw_k = total_cells // smallest_population_size
    bounded_k = max(1, raw_k)
    class_cap_k = max(1, 2 * num_labels)
    return min(bounded_k, labeled_cells, class_cap_k)


def cap_n_jobs_for_k(requested_n_jobs: int, k: int) -> int:
    """Reduce thread count for very large neighborhoods to avoid memory spikes."""
    if requested_n_jobs <= 1:
        return 1
    if k >= 10_000:
        return 1
    if k >= 5_000:
        return min(requested_n_jobs, 2)
    if k >= 2_000:
        return min(requested_n_jobs, 4)
    return requested_n_jobs


def batch_size_for_k(
    k: int,
    default_batch_size: int,
    min_batch_size: int,
    target_index_bytes_per_batch: int,
) -> int:
    """Compute prediction batch size from k and memory target."""
    if k <= 0:
        return default_batch_size
    max_rows = target_index_bytes_per_batch // (k * np.dtype(np.int64).itemsize)
    return max(min_batch_size, min(default_batch_size, int(max_rows)))


def fit_prototype_model(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    requested_n_jobs: int,
    prototypes_per_class: int,
) -> tuple[PrototypeModel, FitStats]:
    """Train an approximate prototype model using per-class mini-batch k-means."""
    if train_matrix.shape[0] != train_labels.shape[0]:
        raise ValueError(
            "Number of labels does not match rows in training matrix "
            f"({train_labels.shape[0]} != {train_matrix.shape[0]})."
        )
    if prototypes_per_class <= 0:
        raise ValueError("prototypes_per_class must be > 0.")

    label_values = train_labels.astype(float, copy=False)
    labeled_mask = ~np.isnan(label_values)
    labeled_labels = label_values[labeled_mask].astype(int)
    if labeled_labels.size == 0:
        raise ValueError("No labeled rows available after excluding unlabeled class 0.")

    x_labeled = np.asarray(train_matrix[labeled_mask], dtype=np.float32, order='C')
    unique_labels, counts = np.unique(labeled_labels, return_counts=True)

    smallest_population_size = int(counts.min())
    num_labels = int(unique_labels.size)
    labeled_cells = int(labeled_labels.size)
    total_cells = int(train_matrix.shape[0])
    k = compute_k(total_cells, smallest_population_size, labeled_cells, num_labels)

    n_jobs_effective = cap_n_jobs_for_k(requested_n_jobs, k)

    centers_list: list[np.ndarray] = []
    label_list: list[int] = []
    for label, count in zip(unique_labels, counts):
        class_rows = x_labeled[labeled_labels == int(label)]
        if class_rows.shape[0] == 0:
            continue
        clusters = min(int(prototypes_per_class), int(count))
        if clusters == 1:
            centers = class_rows.mean(axis=0, keepdims=True)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=clusters,
                random_state=42,
                batch_size=min(16384, class_rows.shape[0]),
                n_init=3,
                max_iter=100,
            )
            kmeans.fit(class_rows)
            centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
        centers_list.append(centers)
        label_list.extend([int(label)] * centers.shape[0])

    if not centers_list:
        raise ValueError("Unable to build prototype model: no prototype centers created.")

    model = PrototypeModel(
        centers=np.vstack(centers_list).astype(np.float32, copy=False),
        labels=np.asarray(label_list, dtype=int),
    )
    stats = FitStats(
        k=k,
        n_jobs_effective=n_jobs_effective,
        prototype_count=int(model.centers.shape[0]),
    )
    return model, stats


def predict_in_batches(
    model: PrototypeModel,
    sample_matrix: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict labels using nearest prototype in fixed-size batches."""
    if batch_size <= 0:
        raise ValueError("Predict batch size must be > 0.")
    total_rows = sample_matrix.shape[0]
    if total_rows == 0:
        return np.array([], dtype=int)

    sample_matrix = np.asarray(sample_matrix, dtype=np.float32, order='C')
    predictions = np.empty(total_rows, dtype=int)
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        nearest_idx = pairwise_distances_argmin(
            sample_matrix[start_idx:end_idx], model.centers, metric='euclidean'
        )
        predictions[start_idx:end_idx] = model.labels[nearest_idx]
    return predictions

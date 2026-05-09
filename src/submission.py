from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import LABEL_COLUMNS


def align_probabilities(
    probabilities: np.ndarray,
    class_order: Sequence[str],
    target_order: Sequence[str] = LABEL_COLUMNS,
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got shape {matrix.shape}")
    if len(class_order) != matrix.shape[1]:
        raise ValueError(
            f"Class order length {len(class_order)} does not match probability columns {matrix.shape[1]}"
        )

    class_to_index = {label: index for index, label in enumerate(class_order)}
    missing = [label for label in target_order if label not in class_to_index]
    if missing:
        raise ValueError(f"Model probabilities are missing classes: {missing}")

    aligned = np.zeros((matrix.shape[0], len(target_order)), dtype=float)
    for target_index, label in enumerate(target_order):
        aligned[:, target_index] = matrix[:, class_to_index[label]]
    return normalize_probabilities(aligned)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    _validate_probability_matrix(matrix, require_unit_sum=False)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def make_submission_frame(ids: Sequence[object], probabilities: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(probabilities, dtype=float)
    _validate_probability_matrix(matrix, require_unit_sum=True)
    if len(ids) != matrix.shape[0]:
        raise ValueError(f"Expected {len(ids)} probability rows, got {matrix.shape[0]}")
    submission = pd.DataFrame(matrix, columns=LABEL_COLUMNS)
    submission.insert(0, "id", list(ids))
    validate_submission_frame(submission)
    return submission


def validate_submission_frame(frame: pd.DataFrame, expected_row_count: int | None = None) -> None:
    expected = ["id", *LABEL_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(f"Submission columns must be exactly {expected}, got {list(frame.columns)}")
    if expected_row_count is not None and len(frame) != expected_row_count:
        raise ValueError(f"Submission must contain {expected_row_count} rows, got {len(frame)}")
    if frame["id"].duplicated().any():
        duplicate_count = int(frame["id"].duplicated().sum())
        raise ValueError(f"Submission contains duplicate ids: {duplicate_count}")
    probabilities = frame[LABEL_COLUMNS].to_numpy(dtype=float)
    if probabilities.shape[0] != len(frame):
        raise ValueError("Submission probability shape does not match row count")
    _validate_probability_matrix(probabilities, require_unit_sum=True)


def write_submission(frame: pd.DataFrame, output_path: str | Path) -> Path:
    validate_submission_frame(frame)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def _validate_probability_matrix(probabilities: np.ndarray, require_unit_sum: bool) -> None:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got shape {matrix.shape}")
    if matrix.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(f"Expected {len(LABEL_COLUMNS)} probability columns, got {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError("Probabilities contain NaN or infinite values")
    if (matrix < 0).any():
        raise ValueError("Probabilities must be non-negative")

    row_sums = matrix.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("Every prediction row must contain at least one positive probability")
    if require_unit_sum and not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Submission probabilities must sum to 1 per row before writing")

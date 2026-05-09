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

    class_to_index = {label: index for index, label in enumerate(class_order)}
    aligned = np.zeros((matrix.shape[0], len(target_order)), dtype=float)
    for target_index, label in enumerate(target_order):
        if label in class_to_index:
            aligned[:, target_index] = matrix[:, class_to_index[label]]
    return normalize_probabilities(aligned)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Probabilities contain NaN or infinite values")
    matrix = np.clip(matrix, 0.0, None)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Every prediction row must contain at least one positive probability")
    return matrix / row_sums


def make_submission_frame(ids: Sequence[object], probabilities: np.ndarray) -> pd.DataFrame:
    normalized = normalize_probabilities(probabilities)
    if len(ids) != normalized.shape[0]:
        raise ValueError(f"Expected {len(ids)} probability rows, got {normalized.shape[0]}")
    submission = pd.DataFrame(normalized, columns=LABEL_COLUMNS)
    submission.insert(0, "id", list(ids))
    validate_submission_frame(submission)
    return submission


def validate_submission_frame(frame: pd.DataFrame) -> None:
    expected = ["id", *LABEL_COLUMNS]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Submission is missing columns: {missing}")
    probabilities = frame[LABEL_COLUMNS].to_numpy(dtype=float)
    if probabilities.shape[0] != len(frame):
        raise ValueError("Submission probability shape does not match row count")
    normalized = normalize_probabilities(probabilities)
    if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Submission probabilities must sum to 1 per row")


def write_submission(frame: pd.DataFrame, output_path: str | Path) -> Path:
    validate_submission_frame(frame)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination

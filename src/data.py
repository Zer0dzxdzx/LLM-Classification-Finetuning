from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_COLUMNS = ["prompt", "response_a", "response_b"]


def read_csv_required(path: str | Path, purpose: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{purpose} file not found: {csv_path}. "
            "In Kaggle, add the competition dataset and use "
            "/kaggle/input/llm-classification-finetuning. "
            "Locally, use configs/local_sample.yaml for a synthetic smoke test."
        )
    return pd.read_csv(csv_path)


def load_training_frame(
    train_path: str | Path,
    sample_size: int | None = None,
    random_seed: int = 42,
    text_max_chars: int | None = None,
) -> pd.DataFrame:
    frame = read_csv_required(train_path, "Training")
    _require_columns(frame, ["id", *TEXT_COLUMNS, *LABEL_COLUMNS], "training")

    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    prepared = pd.DataFrame(
        {
            "id": frame["id"],
            "text": [build_pair_text(row, text_max_chars=text_max_chars) for _, row in frame.iterrows()],
            "target": [target_from_row(row) for _, row in frame.iterrows()],
        }
    )
    return prepared


def load_test_frame(test_path: str | Path, text_max_chars: int | None = None) -> pd.DataFrame:
    frame = read_csv_required(test_path, "Test")
    _require_columns(frame, ["id", *TEXT_COLUMNS], "test")
    return pd.DataFrame(
        {
            "id": frame["id"],
            "text": [build_pair_text(row, text_max_chars=text_max_chars) for _, row in frame.iterrows()],
        }
    )


def build_pair_text(row: pd.Series | dict[str, object], text_max_chars: int | None = None) -> str:
    prompt = normalize_text(row.get("prompt", ""))
    response_a = normalize_text(row.get("response_a", ""))
    response_b = normalize_text(row.get("response_b", ""))
    text = f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"
    if text_max_chars is not None and text_max_chars > 0:
        return text[:text_max_chars]
    return text


def normalize_text(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (normalize_text(item) for item in value) if part)

    text = str(value).strip()
    if _looks_like_literal_sequence(text):
        parsed = _safe_literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return normalize_text(parsed)
    return text


def target_from_row(row: pd.Series | dict[str, object]) -> str:
    winners = [label for label in LABEL_COLUMNS if _as_int(row.get(label, 0)) == 1]
    if len(winners) != 1:
        row_id = row.get("id", "<unknown>")
        raise ValueError(f"Expected exactly one winner label for row id={row_id}, got {winners}")
    return winners[0]


def validate_class_counts(labels: Iterable[str], validation_size: float) -> None:
    counts = pd.Series(list(labels)).value_counts()
    missing = [label for label in LABEL_COLUMNS if label not in counts.index]
    if missing:
        raise ValueError(f"Training data is missing required classes: {missing}")

    if validation_size <= 0:
        return

    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Stratified validation requires at least 2 rows per class; "
            f"small classes: {too_small.to_dict()}"
        )


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {name} data: {missing}")


def _looks_like_literal_sequence(text: str) -> bool:
    return len(text) >= 2 and text[0] in "[(" and text[-1] in "])"


def _safe_literal_eval(text: str) -> object:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _as_int(value: object) -> int:
    if _is_missing(value):
        return 0
    return int(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False

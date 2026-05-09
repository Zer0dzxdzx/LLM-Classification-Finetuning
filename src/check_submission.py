from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import LABEL_COLUMNS
from src.submission import validate_submission_frame


def check_submission(path: str | Path, expected_rows: int | None = None) -> dict[str, object]:
    submission_path = Path(path)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {submission_path}")

    frame = pd.read_csv(submission_path)
    validate_submission_frame(frame, expected_row_count=expected_rows)
    row_sums = frame[LABEL_COLUMNS].sum(axis=1)
    return {
        "path": str(submission_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "probability_sum_min": float(row_sums.min()),
        "probability_sum_max": float(row_sums.max()),
        "duplicate_ids": int(frame["id"].duplicated().sum()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission CSV.")
    parser.add_argument("path", help="Submission CSV path.")
    parser.add_argument("--expected-rows", type=int, help="Optional expected row count.")
    args = parser.parse_args(argv)

    result = check_submission(args.path, expected_rows=args.expected_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

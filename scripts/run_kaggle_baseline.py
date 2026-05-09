from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "submissions" / "submission.csv"
KAGGLE_TRAIN = Path("/kaggle/input/llm-classification-finetuning/train.csv")
LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the first Kaggle baseline end to end.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Baseline YAML config path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Submission CSV output path.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit tests before training.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    output_path = Path(args.output)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    _require_kaggle_data()
    if not args.skip_tests:
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])

    train_result = _run_json([sys.executable, "-m", "src.train", "--config", str(config_path)])
    predict_result = _run_json(
        [
            sys.executable,
            "-m",
            "src.predict",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )
    submission_summary = _summarize_submission(output_path)
    _print_experiment_log_template(config_path, output_path, train_result, predict_result, submission_summary)


def _require_kaggle_data() -> None:
    if not KAGGLE_TRAIN.exists():
        raise FileNotFoundError(
            f"Kaggle training file was not found at {KAGGLE_TRAIN}. "
            "Open this in a Kaggle Notebook and add the competition dataset first."
        )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=True, text=True)


def _run_json(command: list[str]) -> dict[str, object]:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout, end="")
    return json.loads(completed.stdout)


def _summarize_submission(output_path: Path) -> dict[str, object]:
    if not output_path.exists():
        raise FileNotFoundError(f"Submission file was not created: {output_path}")
    submission = pd.read_csv(output_path)
    expected_columns = ["id", *LABEL_COLUMNS]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be {expected_columns}, got {list(submission.columns)}")
    row_sums = submission[LABEL_COLUMNS].sum(axis=1)
    if not ((row_sums - 1.0).abs() < 1e-6).all():
        raise ValueError("Submission probabilities do not sum to 1 for every row.")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicate ids.")
    return {
        "rows": int(len(submission)),
        "columns": expected_columns,
        "probability_sum_min": float(row_sums.min()),
        "probability_sum_max": float(row_sums.max()),
        "path": str(output_path),
    }


def _print_experiment_log_template(
    config_path: Path,
    output_path: Path,
    train_result: dict[str, object],
    predict_result: dict[str, object],
    submission_summary: dict[str, object],
) -> None:
    print("\n=== Baseline run complete ===")
    print(json.dumps({"prediction": predict_result, "submission": submission_summary}, indent=2, ensure_ascii=False))
    print("\nPaste this into docs/experiment_log.md after you submit on Kaggle:\n")
    print(
        "\n".join(
            [
                f"Date: {datetime.now(timezone.utc).date().isoformat()}",
                f"Config: {config_path.relative_to(PROJECT_ROOT)}",
                "Hypothesis: TF-IDF + Logistic Regression can provide the first reliable baseline.",
                "Command:",
                f"  python scripts/run_kaggle_baseline.py --config {config_path.relative_to(PROJECT_ROOT)} --output {output_path.relative_to(PROJECT_ROOT)}",
                f"Validation log loss: {train_result.get('validation_log_loss')}",
                f"Validation accuracy: {train_result.get('validation_accuracy')}",
                "Public leaderboard:",
                "What worked:",
                "What failed:",
                "Next step:",
            ]
        )
    )


if __name__ == "__main__":
    main()

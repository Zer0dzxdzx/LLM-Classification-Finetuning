from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "submissions" / "submission.csv"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
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

    data_dir = _find_kaggle_data_dir()
    runtime_config_path = _write_runtime_config(config_path, data_dir)
    if not args.skip_tests:
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])

    train_result = _run_json([sys.executable, "-m", "src.train", "--config", str(runtime_config_path)])
    predict_result = _run_json(
        [
            sys.executable,
            "-m",
            "src.predict",
            "--config",
            str(runtime_config_path),
            "--output",
            str(output_path),
        ]
    )
    submission_summary = _summarize_submission(output_path)
    _print_experiment_log_template(config_path, output_path, train_result, predict_result, submission_summary)


def _find_kaggle_data_dir(input_root: Path = KAGGLE_INPUT_ROOT) -> Path:
    candidates = sorted(
        path.parent
        for path in input_root.rglob("train.csv")
        if (path.parent / "test.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find train.csv and test.csv under {input_root}. "
            "Open this in a Kaggle Notebook and add the competition dataset first."
        )
    if len(candidates) > 1:
        formatted = "\n".join(f"- {candidate}" for candidate in candidates)
        raise ValueError(
            "Found multiple Kaggle input directories with train.csv and test.csv. "
            "Keep only the competition dataset attached, or pass a custom config.\n"
            f"{formatted}"
        )
    print(f"Using Kaggle data directory: {candidates[0]}")
    return candidates[0]


def _write_runtime_config(config_path: Path, data_dir: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    raw_config.setdefault("data", {})
    raw_config["data"]["train_path"] = str(data_dir / "train.csv")
    raw_config["data"]["test_path"] = str(data_dir / "test.csv")

    runtime_config = PROJECT_ROOT / "outputs" / "runtime_baseline.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    return runtime_config


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

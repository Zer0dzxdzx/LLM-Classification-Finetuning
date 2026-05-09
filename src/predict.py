from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from src.config import ProjectConfig, load_config
from src.data import LABEL_COLUMNS, load_test_frame
from src.submission import align_probabilities, make_submission_frame, write_submission


def run_prediction(config: ProjectConfig, output_path: str | Path) -> dict[str, Any]:
    artifact = load_model_artifact(config.model.artifact_path)
    pipeline = artifact["pipeline"]
    class_order = artifact.get("class_order") or list(pipeline.named_steps["classifier"].classes_)

    test_frame = load_test_frame(config.data.test_path, text_max_chars=config.data.text_max_chars)
    raw_probabilities = pipeline.predict_proba(test_frame["text"])
    probabilities = align_probabilities(raw_probabilities, class_order, LABEL_COLUMNS)
    submission = make_submission_frame(test_frame["id"], probabilities)
    destination = write_submission(submission, output_path)

    return {
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "output_path": str(destination),
    }


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. "
            "Train first with python -m src.train --config <config>."
        )
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid model artifact: {artifact_path}")
    labels = artifact.get("label_columns")
    if labels != LABEL_COLUMNS:
        raise ValueError(f"Artifact label columns {labels} do not match expected {LABEL_COLUMNS}")
    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission file.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_prediction(config, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

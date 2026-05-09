from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from src.baseline import build_baseline_pipeline
from src.config import ProjectConfig, config_to_dict, load_config
from src.data import LABEL_COLUMNS, load_training_frame, validate_class_counts
from src.submission import align_probabilities


def run_training(config: ProjectConfig) -> dict[str, Any]:
    if config.model.type != "tfidf_logreg":
        raise ValueError(f"Unsupported model.type for baseline trainer: {config.model.type}")

    train_frame = load_training_frame(
        config.data.train_path,
        sample_size=config.data.sample_size,
        random_seed=config.train.random_seed,
        text_max_chars=config.data.text_max_chars,
    )
    validate_class_counts(train_frame["target"], config.train.validation_size)

    train_part, valid_part = train_test_split(
        train_frame,
        test_size=config.train.validation_size,
        random_state=config.train.random_seed,
        stratify=train_frame["target"],
    )

    pipeline = build_baseline_pipeline(config.features, config.train)
    pipeline.fit(train_part["text"], train_part["target"])

    raw_probabilities = pipeline.predict_proba(valid_part["text"])
    class_order = list(pipeline.named_steps["classifier"].classes_)
    probabilities = align_probabilities(raw_probabilities, class_order)
    score = log_loss(valid_part["target"], probabilities, labels=LABEL_COLUMNS)
    predictions = pd.Series(probabilities.argmax(axis=1)).map(dict(enumerate(LABEL_COLUMNS)))
    accuracy = accuracy_score(valid_part["target"], predictions)

    artifact = {
        "pipeline": pipeline,
        "label_columns": LABEL_COLUMNS,
        "class_order": class_order,
        "config": config_to_dict(config),
    }
    _save_joblib(artifact, config.model.artifact_path)

    metrics = {
        "model_type": config.model.type,
        "train_rows": int(len(train_part)),
        "valid_rows": int(len(valid_part)),
        "label_columns": LABEL_COLUMNS,
        "classifier_class_order": class_order,
        "validation_log_loss": float(score),
        "validation_accuracy": float(accuracy),
        "artifact_path": str(config.model.artifact_path),
    }
    _write_json(metrics, config.outputs.metrics_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF baseline model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metrics = run_training(config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _save_joblib(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

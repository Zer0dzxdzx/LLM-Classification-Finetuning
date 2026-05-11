from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

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

    if config.train.cv_folds > 1:
        metrics = _run_cv_training(config, train_frame)
    else:
        metrics = _run_holdout_training(config, train_frame)
    _write_json(metrics, config.outputs.metrics_path)
    return metrics


def _run_holdout_training(config: ProjectConfig, train_frame: pd.DataFrame) -> dict[str, Any]:
    train_part, valid_part = train_test_split(
        train_frame,
        test_size=config.train.validation_size,
        random_state=config.train.random_seed,
        stratify=train_frame["target"],
    )

    pipeline = build_baseline_pipeline(config.features, config.train)
    pipeline.fit(train_part, train_part["target"])
    score, accuracy, class_order = _evaluate_pipeline(pipeline, valid_part)

    final_pipeline = build_baseline_pipeline(config.features, config.train)
    final_pipeline.fit(train_frame, train_frame["target"])
    final_class_order = list(final_pipeline.named_steps["classifier"].classes_)
    _save_artifact(config, final_pipeline, final_class_order)

    return {
        "model_type": config.model.type,
        "feature_kind": config.features.kind,
        "use_text_stats": config.features.use_text_stats,
        "validation_strategy": "holdout",
        "train_rows": int(len(train_part)),
        "valid_rows": int(len(valid_part)),
        "fit_rows": int(len(train_frame)),
        "label_columns": LABEL_COLUMNS,
        "classifier_class_order": final_class_order,
        "validation_log_loss": float(score),
        "validation_accuracy": float(accuracy),
        "artifact_path": str(config.model.artifact_path),
    }


def _run_cv_training(config: ProjectConfig, train_frame: pd.DataFrame) -> dict[str, Any]:
    counts = train_frame["target"].value_counts()
    min_class_count = int(counts.min())
    if config.train.cv_folds > min_class_count:
        raise ValueError(
            f"train.cv_folds={config.train.cv_folds} exceeds smallest class count {min_class_count}"
        )

    splitter = StratifiedKFold(
        n_splits=config.train.cv_folds,
        shuffle=True,
        random_state=config.train.random_seed,
    )
    fold_metrics = []
    for fold_index, (train_index, valid_index) in enumerate(
        splitter.split(train_frame, train_frame["target"]),
        start=1,
    ):
        train_part = train_frame.iloc[train_index]
        valid_part = train_frame.iloc[valid_index]
        pipeline = build_baseline_pipeline(config.features, config.train)
        pipeline.fit(train_part, train_part["target"])
        score, accuracy, _class_order = _evaluate_pipeline(pipeline, valid_part)
        fold_metrics.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_part)),
                "valid_rows": int(len(valid_part)),
                "validation_log_loss": float(score),
                "validation_accuracy": float(accuracy),
            }
        )

    final_pipeline = build_baseline_pipeline(config.features, config.train)
    final_pipeline.fit(train_frame, train_frame["target"])
    final_class_order = list(final_pipeline.named_steps["classifier"].classes_)
    _save_artifact(config, final_pipeline, final_class_order)

    total_valid_rows = sum(item["valid_rows"] for item in fold_metrics)
    weighted_log_loss = sum(
        item["validation_log_loss"] * item["valid_rows"] for item in fold_metrics
    ) / total_valid_rows
    weighted_accuracy = sum(
        item["validation_accuracy"] * item["valid_rows"] for item in fold_metrics
    ) / total_valid_rows

    return {
        "model_type": config.model.type,
        "feature_kind": config.features.kind,
        "use_text_stats": config.features.use_text_stats,
        "validation_strategy": "stratified_kfold",
        "cv_folds": config.train.cv_folds,
        "fit_rows": int(len(train_frame)),
        "label_columns": LABEL_COLUMNS,
        "classifier_class_order": final_class_order,
        "validation_log_loss": float(weighted_log_loss),
        "validation_accuracy": float(weighted_accuracy),
        "fold_metrics": fold_metrics,
        "artifact_path": str(config.model.artifact_path),
    }


def _evaluate_pipeline(pipeline: Any, valid_part: pd.DataFrame) -> tuple[float, float, list[str]]:
    raw_probabilities = pipeline.predict_proba(valid_part)
    class_order = list(pipeline.named_steps["classifier"].classes_)
    probabilities = align_probabilities(raw_probabilities, class_order)
    score = log_loss(valid_part["target"], probabilities, labels=LABEL_COLUMNS)
    predictions = pd.Series(probabilities.argmax(axis=1)).map(dict(enumerate(LABEL_COLUMNS)))
    accuracy = accuracy_score(valid_part["target"], predictions)
    return float(score), float(accuracy), class_order


def _save_artifact(config: ProjectConfig, pipeline: Any, class_order: list[str]) -> None:
    artifact = {
        "pipeline": pipeline,
        "label_columns": LABEL_COLUMNS,
        "class_order": class_order,
        "config": config_to_dict(config),
    }
    _save_joblib(artifact, config.model.artifact_path)


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

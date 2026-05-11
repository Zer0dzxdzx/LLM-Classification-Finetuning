from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import LABEL_COLUMNS
from src.predict import run_prediction
from src.predict import predict_probabilities_from_artifact
from src.train import run_training
from tests.helpers import project_path


class WorkflowTests(unittest.TestCase):
    def test_local_sample_training_and_prediction(self) -> None:
        base_config = load_config(project_path("configs", "local_sample.yaml"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = replace(
                base_config,
                model=replace(base_config.model, artifact_path=tmp_path / "model.joblib"),
                outputs=replace(base_config.outputs, metrics_path=tmp_path / "metrics.json"),
            )

            metrics = run_training(config)
            self.assertIn("validation_log_loss", metrics)
            self.assertTrue(config.model.artifact_path.exists())
            self.assertTrue(config.outputs.metrics_path.exists())

            output_path = tmp_path / "submission.csv"
            result = run_prediction(config, output_path)
            self.assertEqual(result["rows"], 3)
            submission = pd.read_csv(output_path)
            self.assertEqual(list(submission.columns), ["id", *LABEL_COLUMNS])
            self.assertTrue(((submission[LABEL_COLUMNS].sum(axis=1) - 1.0).abs() < 1e-6).all())

    def test_local_sample_word_char_stats_cv_workflow(self) -> None:
        base_config = load_config(project_path("configs", "local_sample.yaml"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = replace(
                base_config,
                features=replace(
                    base_config.features,
                    kind="word_char",
                    max_features=200,
                    char_max_features=200,
                    min_df=1,
                    max_df=1.0,
                    use_text_stats=True,
                ),
                train=replace(base_config.train, cv_folds=3, class_weight=None),
                model=replace(base_config.model, artifact_path=tmp_path / "model.joblib"),
                outputs=replace(base_config.outputs, metrics_path=tmp_path / "metrics.json"),
            )

            metrics = run_training(config)
            self.assertEqual(metrics["validation_strategy"], "stratified_kfold")
            self.assertEqual(metrics["cv_folds"], 3)
            self.assertEqual(len(metrics["fold_metrics"]), 3)

            output_path = tmp_path / "submission.csv"
            result = run_prediction(config, output_path)
            self.assertEqual(result["rows"], 3)
            submission = pd.read_csv(output_path)
            self.assertTrue(((submission[LABEL_COLUMNS].sum(axis=1) - 1.0).abs() < 1e-6).all())

    def test_legacy_tfidf_artifact_prediction_uses_text_series(self) -> None:
        train_frame = pd.DataFrame(
            {
                "text": ["a wins", "b wins", "tie case", "a better", "b better", "same"],
                "target": LABEL_COLUMNS * 2,
            }
        )
        legacy_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("classifier", LogisticRegression(max_iter=100, class_weight=None)),
            ]
        )
        legacy_pipeline.fit(train_frame["text"], train_frame["target"])
        test_frame = pd.DataFrame(
            {
                "text": ["a answer", "b answer", "same answer"],
                "prompt_chars": [1.0, 1.0, 1.0],
            }
        )

        probabilities = predict_probabilities_from_artifact(legacy_pipeline, test_frame)

        self.assertEqual(probabilities.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()

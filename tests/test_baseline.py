from __future__ import annotations

import unittest

import pandas as pd

from src.baseline import build_baseline_pipeline
from src.config import FeatureConfig, TrainConfig
from src.data import LABEL_COLUMNS, TEXT_STAT_COLUMNS


class BaselinePipelineTests(unittest.TestCase):
    def test_word_char_stats_pipeline_fits_dataframe(self) -> None:
        frame = pd.DataFrame(
            {
                "text": [
                    "Prompt one Response A good Response B bad",
                    "Prompt two Response A bad Response B good",
                    "Prompt three both similar",
                    "Prompt four A clear B weak",
                    "Prompt five A weak B clear",
                    "Prompt six both similar",
                ],
                "target": LABEL_COLUMNS * 2,
                **{column: [1.0, 2.0, 1.0, 3.0, 2.0, 1.0] for column in TEXT_STAT_COLUMNS},
            }
        )
        features = FeatureConfig(
            kind="word_char",
            max_features=100,
            char_max_features=100,
            min_df=1,
            max_df=1.0,
            use_text_stats=True,
        )
        pipeline = build_baseline_pipeline(features, TrainConfig(class_weight=None, max_iter=100))

        pipeline.fit(frame, frame["target"])
        probabilities = pipeline.predict_proba(frame)

        self.assertEqual(probabilities.shape, (6, 3))


if __name__ == "__main__":
    unittest.main()

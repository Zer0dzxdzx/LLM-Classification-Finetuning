from __future__ import annotations

import unittest

from src.config import make_config, load_config
from tests.helpers import PROJECT_ROOT, project_path


class ConfigTests(unittest.TestCase):
    def test_load_local_sample_config_resolves_paths(self) -> None:
        config = load_config(project_path("configs", "local_sample.yaml"))

        self.assertTrue(config.data.train_path.is_absolute())
        self.assertTrue(config.data.test_path.is_absolute())
        self.assertEqual(config.features.ngram_range, (1, 2))
        self.assertEqual(config.train.random_seed, 42)
        self.assertEqual(config.project_root, PROJECT_ROOT)

    def test_invalid_config_bounds_fail_early(self) -> None:
        raw = {
            "data": {"train_path": "train.csv", "test_path": "test.csv", "sample_size": 0},
            "train": {"validation_size": 1.0},
            "features": {"max_features": -1},
        }

        with self.assertRaisesRegex(ValueError, "data.sample_size"):
            make_config(raw, project_root=PROJECT_ROOT)

        raw["data"]["sample_size"] = 10
        with self.assertRaisesRegex(ValueError, "features.max_features"):
            make_config(raw, project_root=PROJECT_ROOT)

        raw["features"]["max_features"] = 100
        with self.assertRaisesRegex(ValueError, "train.validation_size"):
            make_config(raw, project_root=PROJECT_ROOT)

    def test_load_experiment_configs(self) -> None:
        config_names = [
            "baseline.yaml",
            "tfidf_char.yaml",
            "tfidf_word_char.yaml",
            "tfidf_features.yaml",
            "tfidf_cv.yaml",
        ]

        configs = [load_config(project_path("configs", name)) for name in config_names]

        self.assertEqual(configs[1].features.kind, "char")
        self.assertEqual(configs[2].features.kind, "word_char")
        self.assertTrue(configs[3].features.use_text_stats)
        self.assertEqual(configs[4].train.cv_folds, 3)

    def test_invalid_feature_kind_fails(self) -> None:
        raw = {
            "data": {"train_path": "train.csv", "test_path": "test.csv"},
            "features": {"kind": "unknown"},
        }

        with self.assertRaisesRegex(ValueError, "features.kind"):
            make_config(raw, project_root=PROJECT_ROOT)

    def test_quoted_boolean_values_parse_strictly(self) -> None:
        raw = {
            "data": {"train_path": "train.csv", "test_path": "test.csv"},
            "features": {"lowercase": "false", "use_text_stats": "true"},
        }

        config = make_config(raw, project_root=PROJECT_ROOT)

        self.assertFalse(config.features.lowercase)
        self.assertTrue(config.features.use_text_stats)

        raw["features"]["lowercase"] = "maybe"
        with self.assertRaisesRegex(ValueError, "features.lowercase"):
            make_config(raw, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()

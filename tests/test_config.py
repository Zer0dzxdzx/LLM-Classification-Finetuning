from __future__ import annotations

import unittest
from pathlib import Path

from src.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_local_sample_config_resolves_paths(self) -> None:
        config = load_config("configs/local_sample.yaml")

        self.assertTrue(config.data.train_path.is_absolute())
        self.assertTrue(config.data.test_path.is_absolute())
        self.assertEqual(config.features.ngram_range, (1, 2))
        self.assertEqual(config.train.random_seed, 42)
        self.assertEqual(config.project_root, Path.cwd())


if __name__ == "__main__":
    unittest.main()

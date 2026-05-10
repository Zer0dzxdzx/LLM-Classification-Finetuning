from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.run_kaggle_baseline import (
    _default_output_path,
    _display_path,
    _find_kaggle_data_dir,
    _write_runtime_config,
)
from tests.helpers import project_path


class RunKaggleBaselineTests(unittest.TestCase):
    def test_find_kaggle_data_dir_accepts_variable_input_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            data_dir = input_root / "llm-classification-finetur"
            data_dir.mkdir()
            (data_dir / "train.csv").write_text("id\n1\n", encoding="utf-8")
            (data_dir / "test.csv").write_text("id\n2\n", encoding="utf-8")

            self.assertEqual(_find_kaggle_data_dir(input_root), data_dir)

    def test_find_kaggle_data_dir_fails_without_train_and_test_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            data_dir = input_root / "partial"
            data_dir.mkdir()
            (data_dir / "train.csv").write_text("id\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "train.csv and test.csv"):
                _find_kaggle_data_dir(input_root)

    def test_write_runtime_config_uses_discovered_data_dir(self) -> None:
        config_path = project_path("configs", "baseline.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "competition"
            data_dir.mkdir()
            with patch("scripts.run_kaggle_baseline.PROJECT_ROOT", Path(tmp)):
                runtime_config = _write_runtime_config(config_path, data_dir)

            with runtime_config.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)

            self.assertEqual(config["data"]["train_path"], str(data_dir / "train.csv"))
            self.assertEqual(config["data"]["test_path"], str(data_dir / "test.csv"))

    def test_default_output_uses_kaggle_working_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_working = Path(tmp) / "working"
            fake_working.mkdir()
            self.assertEqual(_default_output_path(fake_working), fake_working / "submission.csv")

    def test_default_output_falls_back_to_project_submissions_locally(self) -> None:
        missing_working = Path("/definitely/not/a/kaggle/working")
        with patch("scripts.run_kaggle_baseline.PROJECT_ROOT", Path("/repo")):
            self.assertEqual(_default_output_path(missing_working), Path("/repo/submissions/submission.csv"))

    def test_display_path_accepts_project_external_output(self) -> None:
        with patch("scripts.run_kaggle_baseline.PROJECT_ROOT", Path("/kaggle/working/repo")):
            self.assertEqual(_display_path(Path("/kaggle/working/repo/configs/baseline.yaml")), "configs/baseline.yaml")
            self.assertEqual(_display_path(Path("/kaggle/working/submission.csv")), "/kaggle/working/submission.csv")


if __name__ == "__main__":
    unittest.main()

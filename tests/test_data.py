from __future__ import annotations

import unittest

import pandas as pd

from src.data import build_pair_text, load_training_frame, normalize_text, target_from_row
from tests.helpers import project_path


class DataTests(unittest.TestCase):
    def test_normalize_text_handles_literal_lists(self) -> None:
        self.assertEqual(normalize_text("['hello', 'world']"), "hello\nworld")
        self.assertEqual(normalize_text("[None, 'answer']"), "answer")

    def test_target_from_row_requires_single_winner(self) -> None:
        row = {"id": 1, "winner_model_a": 0, "winner_model_b": 1, "winner_tie": 0}
        self.assertEqual(target_from_row(row), "winner_model_b")

        with self.assertRaises(ValueError):
            target_from_row({"id": 2, "winner_model_a": 1, "winner_model_b": 1, "winner_tie": 0})

    def test_build_pair_text_uses_expected_sections(self) -> None:
        row = pd.Series(
            {
                "prompt": "Question?",
                "response_a": "Answer A",
                "response_b": "Answer B",
            }
        )
        text = build_pair_text(row)
        self.assertIn("Prompt:\nQuestion?", text)
        self.assertIn("Response A:\nAnswer A", text)
        self.assertIn("Response B:\nAnswer B", text)

    def test_load_training_frame_from_sample(self) -> None:
        frame = load_training_frame(project_path("data", "sample", "train.csv"))
        self.assertEqual(set(frame.columns), {"id", "text", "target"})
        self.assertEqual(len(frame), 9)
        self.assertEqual(set(frame["target"]), {"winner_model_a", "winner_model_b", "winner_tie"})


if __name__ == "__main__":
    unittest.main()

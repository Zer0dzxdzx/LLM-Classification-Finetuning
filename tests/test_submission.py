from __future__ import annotations

import unittest

import numpy as np

from src.data import LABEL_COLUMNS
from src.submission import align_probabilities, make_submission_frame, validate_submission_frame


class SubmissionTests(unittest.TestCase):
    def test_align_probabilities_matches_competition_column_order(self) -> None:
        raw = np.array([[0.2, 0.5, 0.3]])
        aligned = align_probabilities(raw, ["winner_model_b", "winner_tie", "winner_model_a"])

        self.assertTrue(np.allclose(aligned, [[0.3, 0.2, 0.5]]))

    def test_make_submission_frame_validates_shape_and_sum(self) -> None:
        frame = make_submission_frame([101, 102], np.array([[2, 1, 1], [1, 1, 2]], dtype=float))

        self.assertEqual(list(frame.columns), ["id", *LABEL_COLUMNS])
        self.assertTrue(np.allclose(frame[LABEL_COLUMNS].sum(axis=1), 1.0))
        validate_submission_frame(frame)

    def test_zero_probability_rows_fail(self) -> None:
        with self.assertRaises(ValueError):
            make_submission_frame([1], np.array([[0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()

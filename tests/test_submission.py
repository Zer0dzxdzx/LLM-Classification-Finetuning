from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import LABEL_COLUMNS
from src.submission import align_probabilities, make_submission_frame, validate_submission_frame


class SubmissionTests(unittest.TestCase):
    def test_align_probabilities_matches_competition_column_order(self) -> None:
        raw = np.array([[0.2, 0.5, 0.3]])
        aligned = align_probabilities(raw, ["winner_model_b", "winner_tie", "winner_model_a"])

        self.assertTrue(np.allclose(aligned, [[0.3, 0.2, 0.5]]))

    def test_align_probabilities_requires_matching_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing classes"):
            align_probabilities(np.array([[0.5, 0.5]]), ["winner_model_a", "winner_model_b"])

        with self.assertRaisesRegex(ValueError, "length"):
            align_probabilities(np.array([[0.3, 0.3, 0.4]]), ["winner_model_a", "winner_model_b"])

    def test_make_submission_frame_validates_shape_and_sum(self) -> None:
        frame = make_submission_frame([101, 102], np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]))

        self.assertEqual(list(frame.columns), ["id", *LABEL_COLUMNS])
        self.assertTrue(np.allclose(frame[LABEL_COLUMNS].sum(axis=1), 1.0))
        validate_submission_frame(frame)

    def test_invalid_submission_probabilities_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            make_submission_frame([1], np.array([[2.0, 1.0, 1.0]]))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_submission_frame([1], np.array([[1.1, -0.1, 0.0]]))

    def test_zero_probability_rows_fail(self) -> None:
        with self.assertRaises(ValueError):
            make_submission_frame([1], np.array([[0.0, 0.0, 0.0]]))

    def test_submission_requires_exact_columns_and_unique_ids(self) -> None:
        valid = make_submission_frame([1, 2], np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4]]))

        extra_column = valid.assign(extra=0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_submission_frame(extra_column)

        duplicates = pd.concat([valid.iloc[[0]], valid.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            validate_submission_frame(duplicates)

        with self.assertRaisesRegex(ValueError, "3 rows"):
            validate_submission_frame(valid, expected_row_count=3)


if __name__ == "__main__":
    unittest.main()

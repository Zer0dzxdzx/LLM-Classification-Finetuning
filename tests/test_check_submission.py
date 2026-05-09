from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.check_submission import check_submission
from src.data import LABEL_COLUMNS


class CheckSubmissionTests(unittest.TestCase):
    def test_check_submission_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            pd.DataFrame(
                {
                    "id": [1, 2],
                    LABEL_COLUMNS[0]: [0.6, 0.2],
                    LABEL_COLUMNS[1]: [0.3, 0.5],
                    LABEL_COLUMNS[2]: [0.1, 0.3],
                }
            ).to_csv(path, index=False)

            result = check_submission(path, expected_rows=2)

            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["duplicate_ids"], 0)
            self.assertEqual(result["columns"], ["id", *LABEL_COLUMNS])

    def test_check_submission_rejects_bad_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            pd.DataFrame(
                {
                    "id": [1],
                    LABEL_COLUMNS[0]: [0.6],
                    LABEL_COLUMNS[1]: [0.3],
                    LABEL_COLUMNS[2]: [0.1],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "2 rows"):
                check_submission(path, expected_rows=2)


if __name__ == "__main__":
    unittest.main()

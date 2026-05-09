from __future__ import annotations

import json
import unittest

from tests.helpers import project_path


class NotebookHygieneTests(unittest.TestCase):
    def test_notebooks_do_not_commit_outputs(self) -> None:
        for notebook_path in project_path("notebooks").glob("*.ipynb"):
            with self.subTest(notebook=notebook_path.name):
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                for cell in notebook["cells"]:
                    self.assertEqual(cell.get("outputs", []), [])
                    self.assertIsNone(cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()

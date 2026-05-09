# Kaggle Baseline Runbook

Use this runbook for the first real-data submission.

## 1. Create the Notebook

1. Open the **LLM Classification Finetuning** competition on Kaggle.
2. Create a new Notebook.
3. Add the competition dataset from the right sidebar.
4. Confirm the data path:

```python
from pathlib import Path
Path("/kaggle/input/llm-classification-finetuning/train.csv").exists()
```

The output must be `True`.

## 2. Clone and Check the Project

Run these cells in Kaggle:

```bash
git clone https://github.com/Zer0dzxdzx/LLM-Classification-Finetuning.git
cd LLM-Classification-Finetuning
python -m unittest discover -s tests
```

If an import fails, install the core dependencies:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
```

## 3. Run the Baseline

Preferred one-command path:

```bash
python scripts/run_kaggle_baseline.py
```

The script auto-detects the mounted Kaggle input directory by finding the folder that contains both `train.csv` and `test.csv`. In Kaggle, it writes the required code-competition output to:

```text
/kaggle/working/submission.csv
```

Equivalent manual path:

```bash
python -m src.train --config configs/baseline.yaml
python -m src.predict --config configs/baseline.yaml --output /kaggle/working/submission.csv
python -m src.check_submission /kaggle/working/submission.csv
```

## 4. Verify and Submit

Confirm the generated file:

```python
import pandas as pd
submission = pd.read_csv("/kaggle/working/submission.csv")
submission.head()
submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).describe()
```

Save a full Notebook version. Kaggle will privately rerun the selected version and look for `/kaggle/working/submission.csv`.

## 5. Record the Result

After Kaggle returns the public leaderboard score, copy the template printed by `scripts/run_kaggle_baseline.py` into `docs/experiment_log.md` and fill in:

- Public leaderboard
- What worked
- What failed
- Next step

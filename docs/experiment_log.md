# Experiment Log

Use one entry per run. Keep entries short and honest.

## Template

```text
Date:
Config:
Hypothesis:
Command:
Validation log loss:
Public leaderboard:
What worked:
What failed:
Next step:
```

## Runs

### 2026-05-10 - TF-IDF Logistic Regression Baseline

Date: 2026-05-10
Config: Kaggle no-internet notebook cell equivalent to `configs/baseline.yaml`
Hypothesis: TF-IDF + Logistic Regression can provide the first reliable baseline.
Command:
  - Saved a Kaggle Notebook version with internet disabled.
  - Wrote `/kaggle/working/submission.csv`.
Validation log loss: 1.1595351634214317
Validation accuracy: 0.3736951983298539
Public leaderboard: 1.14756
What worked:
  - First Kaggle submission was accepted and scored.
  - The generated submission had shape `(3, 4)` and probability rows summed to 1.
  - The no-internet Notebook path satisfied the competition submission rules.
What failed:
  - GitHub clone based Notebook versions could not be submitted because the competition disallows internet access.
  - Earlier generated submissions were nested under the repository directory instead of `/kaggle/working/submission.csv`.
Next step:
  - Use this score as the baseline for small TF-IDF feature experiments before Transformer finetuning.

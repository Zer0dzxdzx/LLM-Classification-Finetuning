# Two-Week Learning Roadmap

## Week 1: Build the Loop

Day 1:
- Read the competition overview and evaluation metric.
- Inspect `train.csv`, `test.csv`, and sample submission format.
- Run `notebooks/01_eda.ipynb`.

Day 2:
- Study the label definition: `winner_model_a`, `winner_model_b`, `winner_tie`.
- Inspect class balance, missing text, text length, and duplicate prompts.
- Write observations in `docs/experiment_log.md`.

Day 3:
- Run the local sample workflow.
- Train the TF-IDF baseline in Kaggle Notebook.
- Save validation log loss and first submission.

Day 4:
- Improve text construction and TF-IDF parameters.
- Compare word n-grams, character n-grams, truncation, and class weights.

Day 5:
- Review validation errors.
- Manually inspect cases where the baseline is confidently wrong.

## Week 2: Finetune and Reflect

Day 6-7:
- Run a small Transformer finetune with `configs/finetune.yaml`.
- Keep the first run small enough to finish quickly.

Day 8-9:
- Tune sequence length, sample size, learning rate, and batch size.
- Compare finetuned validation log loss with TF-IDF baseline.

Day 10:
- Generate a clean `submission.csv`.
- Record final assumptions, validation score, and leaderboard score.

Day 11-14:
- Optional improvements: model choice, ensembling, calibration, better cross-validation.
- Write a concise postmortem: what worked, what failed, what to learn next.

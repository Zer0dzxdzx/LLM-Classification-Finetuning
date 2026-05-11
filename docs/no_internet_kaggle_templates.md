# No-Internet Kaggle Templates

This competition rejects Notebook versions with internet enabled. Use one of these options.

## Repository Script Path

This works for local development or Kaggle sessions before submission, but not for final no-internet reruns if the Notebook depends on `git clone`.

```bash
python scripts/run_kaggle_baseline.py --skip-tests --output /kaggle/working/submission.csv
```

## Pasteable Notebook Cell

For final submissions, paste the contents of:

```text
kaggle_cells/sklearn_tfidf_experiments.py
```

into one Kaggle code cell, disable internet, save a full version, and submit the successful version.

Change only the `EXPERIMENT` dictionary at the top of the cell. Recommended sequence:

1. `word_baseline`
2. `char_tfidf`
3. `word_char`
4. `word_char_stats`

Every template writes:

```text
/kaggle/working/submission.csv
```

and checks that probability rows sum to 1.

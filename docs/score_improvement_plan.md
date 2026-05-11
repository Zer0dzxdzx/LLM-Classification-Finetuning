# Score Improvement Plan

Current baseline:
- Validation log loss: `1.1595351634214317`
- Validation accuracy: `0.3736951983298539`
- Public leaderboard: `1.14756`

## Learning Focus

1. Understand the target:
   - Three mutually exclusive classes: `winner_model_a`, `winner_model_b`, `winner_tie`.
   - Metric is log loss, so calibrated probabilities matter more than hard labels.
2. Understand the data:
   - Compare text length, empty responses, equal responses, and tie frequency.
   - Inspect cases where A/B are similar but one is preferred.
3. Build traditional NLP strength before Transformer finetuning:
   - Word TF-IDF learns broad content.
   - Character TF-IDF catches style, formatting, punctuation, and short phrases.
   - Length/stat features can expose response asymmetry.
4. Use validation before leaderboard:
   - Public LB is useful but noisy.
   - Prefer 3-fold CV when comparing close variants.

## Experiment Order

Run these in Kaggle with internet disabled and record every result in `docs/experiment_log.md`.

| Order | Config | Goal |
| --- | --- | --- |
| 1 | `configs/baseline.yaml` | Reproduce the accepted baseline. |
| 2 | `configs/tfidf_char.yaml` | Test whether style/format ngrams beat word ngrams. |
| 3 | `configs/tfidf_word_char.yaml` | Combine word and character evidence. |
| 4 | `configs/tfidf_features.yaml` | Add simple response length and equality features. |
| 5 | `configs/tfidf_cv.yaml` | Use 3-fold CV to choose a stable candidate. |

## Decision Rules

- Submit only when validation log loss improves or when a CV result is clearly more stable.
- Treat public leaderboard changes smaller than `0.01` as weak evidence unless validation also improves.
- Keep each Kaggle submission description short and specific, for example `word_char_tfidf_stats`.
- After two or three traditional NLP improvements, revisit Transformer finetuning.

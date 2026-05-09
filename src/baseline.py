from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import FeatureConfig, TrainConfig


def build_baseline_pipeline(features: FeatureConfig, train: TrainConfig) -> Pipeline:
    classifier = LogisticRegression(
        C=train.c,
        class_weight=train.class_weight,
        max_iter=train.max_iter,
        random_state=train.random_seed,
        solver=train.solver,
    )
    vectorizer = TfidfVectorizer(
        lowercase=features.lowercase,
        max_df=features.max_df,
        max_features=features.max_features,
        min_df=features.min_df,
        ngram_range=features.ngram_range,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )

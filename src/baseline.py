from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FeatureConfig, TrainConfig
from src.data import TEXT_STAT_COLUMNS


def build_baseline_pipeline(features: FeatureConfig, train: TrainConfig) -> Pipeline:
    classifier = LogisticRegression(
        C=train.c,
        class_weight=train.class_weight,
        max_iter=train.max_iter,
        random_state=train.random_seed,
        solver=train.solver,
    )
    transformers = []
    if features.kind in {"word", "word_char"}:
        transformers.append(
            (
                "word_tfidf",
                _tfidf_vectorizer(features, analyzer="word"),
                "text",
            )
        )
    if features.kind in {"char", "word_char"}:
        transformers.append(
            (
                "char_tfidf",
                _tfidf_vectorizer(features, analyzer="char_wb"),
                "text",
            )
        )
    if features.use_text_stats:
        transformers.append(("text_stats", StandardScaler(with_mean=False), TEXT_STAT_COLUMNS))

    feature_builder = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    return Pipeline(
        steps=[
            ("features", feature_builder),
            ("classifier", classifier),
        ]
    )


def _tfidf_vectorizer(features: FeatureConfig, analyzer: str) -> TfidfVectorizer:
    ngram_range = features.char_ngram_range if analyzer == "char_wb" else features.ngram_range
    max_features = features.char_max_features if analyzer == "char_wb" else features.max_features
    return TfidfVectorizer(
        analyzer=analyzer,
        lowercase=features.lowercase,
        max_df=features.max_df,
        max_features=max_features or features.max_features,
        min_df=features.min_df,
        ngram_range=ngram_range,
        strip_accents="unicode",
        sublinear_tf=True,
    )

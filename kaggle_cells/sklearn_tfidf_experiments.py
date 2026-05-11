from pathlib import Path
import ast
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=SyntaxWarning)

EXPERIMENT = {
    "name": "word_char_stats",
    "kind": "word_char",
    "max_features": 100000,
    "ngram_range": (1, 2),
    "char_max_features": 100000,
    "char_ngram_range": (3, 5),
    "min_df": 2,
    "max_df": 0.95,
    "use_text_stats": True,
    "c": 2.0,
    "text_max_chars": 12000,
    "validation_size": 0.15,
    "cv_folds": 1,
    "random_seed": 42,
}

LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_STAT_COLUMNS = [
    "prompt_chars",
    "response_a_chars",
    "response_b_chars",
    "response_char_delta",
    "response_char_abs_delta",
    "response_char_ratio",
    "responses_equal",
    "response_a_empty",
    "response_b_empty",
]


def find_data_dir():
    candidates = sorted(
        path.parent
        for path in Path("/kaggle/input").rglob("train.csv")
        if (path.parent / "test.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError("Could not find train.csv and test.csv under /kaggle/input")
    print("Using data dir:", candidates[0])
    return candidates[0]


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (normalize_text(item) for item in value) if part)
    text = str(value).strip()
    if len(text) >= 2 and text[0] in "[(" and text[-1] in "])":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, (list, tuple)):
            return normalize_text(parsed)
    return text


def build_text(row, max_chars):
    text = (
        f"Prompt:\n{normalize_text(row['prompt'])}\n\n"
        f"Response A:\n{normalize_text(row['response_a'])}\n\n"
        f"Response B:\n{normalize_text(row['response_b'])}"
    )
    return text[:max_chars]


def text_stats(row):
    prompt = normalize_text(row["prompt"])
    response_a = normalize_text(row["response_a"])
    response_b = normalize_text(row["response_b"])
    prompt_chars = len(prompt)
    response_a_chars = len(response_a)
    response_b_chars = len(response_b)
    delta = response_a_chars - response_b_chars
    return {
        "prompt_chars": float(prompt_chars),
        "response_a_chars": float(response_a_chars),
        "response_b_chars": float(response_b_chars),
        "response_char_delta": float(delta),
        "response_char_abs_delta": float(abs(delta)),
        "response_char_ratio": float(response_a_chars / max(response_b_chars, 1)),
        "responses_equal": float(response_a == response_b),
        "response_a_empty": float(response_a_chars == 0),
        "response_b_empty": float(response_b_chars == 0),
    }


def target_from_row(row):
    winners = [label for label in LABEL_COLUMNS if int(row[label]) == 1]
    if len(winners) != 1:
        raise ValueError(f"Bad labels for id={row.get('id')}: {winners}")
    return winners[0]


def make_frame(raw, include_target):
    rows = []
    for _, row in raw.iterrows():
        item = {
            "id": row["id"],
            "text": build_text(row, EXPERIMENT["text_max_chars"]),
            **text_stats(row),
        }
        if include_target:
            item["target"] = target_from_row(row)
        rows.append(item)
    return pd.DataFrame(rows)


def tfidf_vectorizer(analyzer):
    return TfidfVectorizer(
        analyzer=analyzer,
        lowercase=True,
        max_df=EXPERIMENT["max_df"],
        max_features=(
            EXPERIMENT["char_max_features"]
            if analyzer == "char_wb"
            else EXPERIMENT["max_features"]
        ),
        min_df=EXPERIMENT["min_df"],
        ngram_range=(
            EXPERIMENT["char_ngram_range"]
            if analyzer == "char_wb"
            else EXPERIMENT["ngram_range"]
        ),
        strip_accents="unicode",
        sublinear_tf=True,
    )


def build_pipeline():
    transformers = []
    if EXPERIMENT["kind"] in {"word", "word_char"}:
        transformers.append(("word_tfidf", tfidf_vectorizer("word"), "text"))
    if EXPERIMENT["kind"] in {"char", "word_char"}:
        transformers.append(("char_tfidf", tfidf_vectorizer("char_wb"), "text"))
    if EXPERIMENT["use_text_stats"]:
        transformers.append(("text_stats", StandardScaler(with_mean=False), TEXT_STAT_COLUMNS))

    return Pipeline(
        steps=[
            ("features", ColumnTransformer(transformers=transformers, sparse_threshold=0.3)),
            (
                "classifier",
                LogisticRegression(
                    C=EXPERIMENT["c"],
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=EXPERIMENT["random_seed"],
                ),
            ),
        ]
    )


def align_probabilities(probabilities, class_order):
    aligned = np.zeros((probabilities.shape[0], len(LABEL_COLUMNS)), dtype=float)
    for index, label in enumerate(LABEL_COLUMNS):
        aligned[:, index] = probabilities[:, class_order.index(label)]
    return aligned / aligned.sum(axis=1, keepdims=True)


data_dir = find_data_dir()
train_raw = pd.read_csv(data_dir / "train.csv")
test_raw = pd.read_csv(data_dir / "test.csv")
train_df = make_frame(train_raw, include_target=True)
test_df = make_frame(test_raw, include_target=False)

print("Experiment:", EXPERIMENT["name"])
if EXPERIMENT["cv_folds"] > 1:
    splitter = StratifiedKFold(
        n_splits=EXPERIMENT["cv_folds"],
        shuffle=True,
        random_state=EXPERIMENT["random_seed"],
    )
    fold_metrics = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(train_df, train_df["target"]),
        start=1,
    ):
        train_part = train_df.iloc[train_index]
        valid_part = train_df.iloc[valid_index]
        pipeline = build_pipeline()
        pipeline.fit(train_part, train_part["target"])
        valid_raw = pipeline.predict_proba(valid_part)
        valid_classes = list(pipeline.named_steps["classifier"].classes_)
        valid_probs = align_probabilities(valid_raw, valid_classes)
        valid_predictions = pd.Series(valid_probs.argmax(axis=1)).map(dict(enumerate(LABEL_COLUMNS)))
        fold_metrics.append(
            {
                "fold": fold,
                "valid_rows": len(valid_part),
                "log_loss": log_loss(valid_part["target"], valid_probs, labels=LABEL_COLUMNS),
                "accuracy": accuracy_score(valid_part["target"], valid_predictions),
            }
        )
        print(fold_metrics[-1])
    total_rows = sum(item["valid_rows"] for item in fold_metrics)
    validation_log_loss = sum(item["log_loss"] * item["valid_rows"] for item in fold_metrics) / total_rows
    validation_accuracy = sum(item["accuracy"] * item["valid_rows"] for item in fold_metrics) / total_rows
else:
    train_part, valid_part = train_test_split(
        train_df,
        test_size=EXPERIMENT["validation_size"],
        random_state=EXPERIMENT["random_seed"],
        stratify=train_df["target"],
    )
    pipeline = build_pipeline()
    pipeline.fit(train_part, train_part["target"])
    valid_raw = pipeline.predict_proba(valid_part)
    valid_classes = list(pipeline.named_steps["classifier"].classes_)
    valid_probs = align_probabilities(valid_raw, valid_classes)
    valid_predictions = pd.Series(valid_probs.argmax(axis=1)).map(dict(enumerate(LABEL_COLUMNS)))
    validation_log_loss = log_loss(valid_part["target"], valid_probs, labels=LABEL_COLUMNS)
    validation_accuracy = accuracy_score(valid_part["target"], valid_predictions)

print("Validation log loss:", validation_log_loss)
print("Validation accuracy:", validation_accuracy)

final_pipeline = build_pipeline()
final_pipeline.fit(train_df, train_df["target"])
test_raw_probs = final_pipeline.predict_proba(test_df)
test_classes = list(final_pipeline.named_steps["classifier"].classes_)
test_probs = align_probabilities(test_raw_probs, test_classes)

submission = pd.DataFrame(test_probs, columns=LABEL_COLUMNS)
submission.insert(0, "id", test_df["id"])
row_sums = submission[LABEL_COLUMNS].sum(axis=1)
assert ((row_sums - 1.0).abs() < 1e-6).all(), row_sums.describe()

output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)
print("Saved:", output_path)
print(submission.shape)
display(submission.head())
print(row_sums.describe())

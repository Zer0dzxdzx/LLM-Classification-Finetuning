from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    test_path: Path
    sample_size: int | None = None
    text_max_chars: int | None = None


@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "word"
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    char_max_features: int | None = None
    min_df: int | float = 2
    max_df: int | float = 0.95
    lowercase: bool = True
    use_text_stats: bool = False


@dataclass(frozen=True)
class ModelConfig:
    type: str = "tfidf_logreg"
    artifact_path: Path = Path("outputs/models/baseline_tfidf_logreg.joblib")


@dataclass(frozen=True)
class TrainConfig:
    validation_size: float = 0.15
    cv_folds: int = 1
    random_seed: int = 42
    max_iter: int = 1000
    solver: str = "lbfgs"
    class_weight: str | dict[str, float] | None = "balanced"
    c: float = 1.0


@dataclass(frozen=True)
class OutputConfig:
    metrics_path: Path = Path("outputs/metrics/metrics.json")


@dataclass(frozen=True)
class FinetuneConfig:
    model_name_or_path: str = "distilbert-base-uncased"
    output_dir: Path = Path("outputs/finetune/distilbert_baseline")
    max_length: int = 512
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    num_train_epochs: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig
    outputs: OutputConfig
    finetune: FinetuneConfig | None = None
    project_root: Path = Path(".")


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return make_config(raw, project_root=project_root)


def make_config(raw: dict[str, Any], project_root: str | Path | None = None) -> ProjectConfig:
    root = Path(project_root or ".").expanduser().resolve()
    data_raw = _section(raw, "data")
    train_raw = _section(raw, "train", required=False)
    feature_raw = _section(raw, "features", required=False)
    model_raw = _section(raw, "model", required=False)
    output_raw = _section(raw, "outputs", required=False)
    finetune_raw = raw.get("finetune")

    data = DataConfig(
        train_path=_resolve_path(_required(data_raw, "train_path"), root),
        test_path=_resolve_path(_required(data_raw, "test_path"), root),
        sample_size=_optional_positive_int(data_raw.get("sample_size"), "data.sample_size"),
        text_max_chars=_optional_positive_int(data_raw.get("text_max_chars"), "data.text_max_chars"),
    )

    features = FeatureConfig(
        kind=_feature_kind(feature_raw.get("kind", feature_raw.get("type", "word"))),
        max_features=_positive_int(feature_raw.get("max_features", 50000), "features.max_features"),
        ngram_range=_ngram_range(feature_raw.get("ngram_range", [1, 2])),
        char_ngram_range=_ngram_range(feature_raw.get("char_ngram_range", [3, 5])),
        char_max_features=_optional_positive_int(feature_raw.get("char_max_features"), "features.char_max_features"),
        min_df=_document_frequency(feature_raw.get("min_df", 2), "features.min_df"),
        max_df=_document_frequency(feature_raw.get("max_df", 0.95), "features.max_df"),
        lowercase=bool(feature_raw.get("lowercase", True)),
        use_text_stats=bool(feature_raw.get("use_text_stats", False)),
    )
    _validate_document_frequency_range(features.min_df, features.max_df)

    model = ModelConfig(
        type=str(model_raw.get("type", "tfidf_logreg")),
        artifact_path=_resolve_path(model_raw.get("artifact_path", "outputs/models/model.joblib"), root),
    )

    train = TrainConfig(
        validation_size=_ratio(train_raw.get("validation_size", 0.15), "train.validation_size", inclusive_upper=False),
        cv_folds=_positive_int(train_raw.get("cv_folds", 1), "train.cv_folds"),
        random_seed=_non_negative_int(train_raw.get("random_seed", 42), "train.random_seed"),
        max_iter=_positive_int(train_raw.get("max_iter", 1000), "train.max_iter"),
        solver=str(train_raw.get("solver", "lbfgs")),
        class_weight=_class_weight(train_raw.get("class_weight", "balanced")),
        c=_positive_float(train_raw.get("c", 1.0), "train.c"),
    )

    outputs = OutputConfig(
        metrics_path=_resolve_path(output_raw.get("metrics_path", "outputs/metrics/metrics.json"), root)
    )

    finetune = None
    if isinstance(finetune_raw, dict):
        finetune = FinetuneConfig(
            model_name_or_path=str(finetune_raw.get("model_name_or_path", "distilbert-base-uncased")),
            output_dir=_resolve_path(finetune_raw.get("output_dir", "outputs/finetune/model"), root),
            max_length=_positive_int(finetune_raw.get("max_length", 512), "finetune.max_length"),
            learning_rate=_positive_float(finetune_raw.get("learning_rate", 2e-5), "finetune.learning_rate"),
            train_batch_size=_positive_int(finetune_raw.get("train_batch_size", 8), "finetune.train_batch_size"),
            eval_batch_size=_positive_int(finetune_raw.get("eval_batch_size", 16), "finetune.eval_batch_size"),
            num_train_epochs=_positive_float(finetune_raw.get("num_train_epochs", 1), "finetune.num_train_epochs"),
            weight_decay=_non_negative_float(finetune_raw.get("weight_decay", 0.01), "finetune.weight_decay"),
            warmup_ratio=_ratio(finetune_raw.get("warmup_ratio", 0.05), "finetune.warmup_ratio"),
            gradient_accumulation_steps=_positive_int(
                finetune_raw.get("gradient_accumulation_steps", 1),
                "finetune.gradient_accumulation_steps",
            ),
        )

    return ProjectConfig(
        data=data,
        features=features,
        model=model,
        train=train,
        outputs=outputs,
        finetune=finetune,
        project_root=root,
    )


def config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    return _to_jsonable(asdict(config))


def _section(raw: dict[str, Any], name: str, required: bool = True) -> dict[str, Any]:
    section = raw.get(name)
    if section is None:
        if required:
            raise KeyError(f"Missing required config section: {name}")
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return section


def _required(section: dict[str, Any], key: str) -> Any:
    value = section.get(key)
    if value in (None, ""):
        raise KeyError(f"Missing required config value: {key}")
    return value


def _resolve_path(value: Any, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    return _positive_int(value, name)


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return number


def _ratio(value: Any, name: str, inclusive_upper: bool = True) -> float:
    number = float(value)
    upper_ok = number <= 1.0 if inclusive_upper else number < 1.0
    if number < 0.0 or not upper_ok:
        boundary = "[0, 1]" if inclusive_upper else "[0, 1)"
        raise ValueError(f"{name} must be in the range {boundary}")
    if name == "train.validation_size" and number == 0.0:
        raise ValueError("train.validation_size must be greater than 0 for stratified validation")
    return number


def _document_frequency(value: Any, name: str) -> int | float:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} as an integer must be at least 1")
        return value

    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0")
    if number <= 1:
        return number
    if number.is_integer():
        return int(number)
    raise ValueError(f"{name} must be an integer count or a float in (0, 1]")


def _validate_document_frequency_range(min_df: int | float, max_df: int | float) -> None:
    if isinstance(min_df, float) and isinstance(max_df, float) and min_df > max_df:
        raise ValueError("features.min_df cannot be greater than features.max_df when both are ratios")


def _class_weight(value: Any) -> str | dict[str, float] | None:
    if value in (None, ""):
        return None
    if value == "balanced" or isinstance(value, dict):
        return value
    raise ValueError("train.class_weight must be null, 'balanced', or a class-to-weight mapping")


def _feature_kind(value: Any) -> str:
    kind = str(value)
    allowed = {"word", "char", "word_char"}
    if kind not in allowed:
        raise ValueError(f"features.kind must be one of {sorted(allowed)}, got {kind}")
    return kind


def _ngram_range(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("features.ngram_range must contain exactly two integers")
    low, high = int(value[0]), int(value[1])
    if low < 1 or high < low:
        raise ValueError("features.ngram_range must satisfy 1 <= low <= high")
    return (low, high)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value

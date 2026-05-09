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
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.95
    lowercase: bool = True


@dataclass(frozen=True)
class ModelConfig:
    type: str = "tfidf_logreg"
    artifact_path: Path = Path("outputs/models/baseline_tfidf_logreg.joblib")


@dataclass(frozen=True)
class TrainConfig:
    validation_size: float = 0.15
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
        sample_size=_optional_int(data_raw.get("sample_size")),
        text_max_chars=_optional_int(data_raw.get("text_max_chars")),
    )

    features = FeatureConfig(
        max_features=int(feature_raw.get("max_features", 50000)),
        ngram_range=_ngram_range(feature_raw.get("ngram_range", [1, 2])),
        min_df=feature_raw.get("min_df", 2),
        max_df=feature_raw.get("max_df", 0.95),
        lowercase=bool(feature_raw.get("lowercase", True)),
    )

    model = ModelConfig(
        type=str(model_raw.get("type", "tfidf_logreg")),
        artifact_path=_resolve_path(model_raw.get("artifact_path", "outputs/models/model.joblib"), root),
    )

    train = TrainConfig(
        validation_size=float(train_raw.get("validation_size", 0.15)),
        random_seed=int(train_raw.get("random_seed", 42)),
        max_iter=int(train_raw.get("max_iter", 1000)),
        solver=str(train_raw.get("solver", "lbfgs")),
        class_weight=train_raw.get("class_weight", "balanced"),
        c=float(train_raw.get("c", 1.0)),
    )

    outputs = OutputConfig(
        metrics_path=_resolve_path(output_raw.get("metrics_path", "outputs/metrics/metrics.json"), root)
    )

    finetune = None
    if isinstance(finetune_raw, dict):
        finetune = FinetuneConfig(
            model_name_or_path=str(finetune_raw.get("model_name_or_path", "distilbert-base-uncased")),
            output_dir=_resolve_path(finetune_raw.get("output_dir", "outputs/finetune/model"), root),
            max_length=int(finetune_raw.get("max_length", 512)),
            learning_rate=float(finetune_raw.get("learning_rate", 2e-5)),
            train_batch_size=int(finetune_raw.get("train_batch_size", 8)),
            eval_batch_size=int(finetune_raw.get("eval_batch_size", 16)),
            num_train_epochs=float(finetune_raw.get("num_train_epochs", 1)),
            weight_decay=float(finetune_raw.get("weight_decay", 0.01)),
            warmup_ratio=float(finetune_raw.get("warmup_ratio", 0.05)),
            gradient_accumulation_steps=int(finetune_raw.get("gradient_accumulation_steps", 1)),
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

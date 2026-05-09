from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from src.config import ProjectConfig, load_config
from src.data import LABEL_COLUMNS, load_training_frame, validate_class_counts


LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_COLUMNS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def run_finetuning(config: ProjectConfig) -> dict[str, Any]:
    if config.finetune is None:
        raise ValueError("Missing finetune section in config.")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Transformer finetuning requires optional dependencies. "
            "Install them with: pip install -r requirements-finetune.txt"
        ) from exc

    frame = load_training_frame(
        config.data.train_path,
        sample_size=config.data.sample_size,
        random_seed=config.train.random_seed,
        text_max_chars=config.data.text_max_chars,
    )
    validate_class_counts(frame["target"], config.train.validation_size)
    train_part, valid_part = train_test_split(
        frame,
        test_size=config.train.validation_size,
        random_state=config.train.random_seed,
        stratify=frame["target"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config.finetune.model_name_or_path)
    train_dataset = TextClassificationDataset(
        texts=train_part["text"].tolist(),
        labels=[LABEL_TO_ID[label] for label in train_part["target"]],
        tokenizer=tokenizer,
        max_length=config.finetune.max_length,
        torch_module=torch,
    )
    valid_dataset = TextClassificationDataset(
        texts=valid_part["text"].tolist(),
        labels=[LABEL_TO_ID[label] for label in valid_part["target"]],
        tokenizer=tokenizer,
        max_length=config.finetune.max_length,
        torch_module=torch,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.finetune.model_name_or_path,
        num_labels=len(LABEL_COLUMNS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = _training_args(config, TrainingArguments)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    config.finetune.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(config.finetune.output_dir)
    tokenizer.save_pretrained(config.finetune.output_dir)

    metrics = {
        "model_name_or_path": config.finetune.model_name_or_path,
        "train_rows": int(len(train_part)),
        "valid_rows": int(len(valid_part)),
        "output_dir": str(config.finetune.output_dir),
        **{key: _json_number(value) for key, value in eval_metrics.items()},
    }
    _write_json(metrics, config.outputs.metrics_path)
    return metrics


class TextClassificationDataset:
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int, torch_module: Any) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.torch = torch_module

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = self.torch.tensor(self.labels[index], dtype=self.torch.long)
        return item


def _compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, labels = eval_prediction
    probabilities = _softmax(np.asarray(logits))
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=list(range(len(LABEL_COLUMNS))))),
        "accuracy": float(accuracy_score(labels, probabilities.argmax(axis=1))),
    }


def _training_args(config: ProjectConfig, training_args_cls: Any) -> Any:
    ft = config.finetune
    if ft is None:
        raise ValueError("Missing finetune section in config.")

    kwargs = {
        "output_dir": str(ft.output_dir),
        "learning_rate": ft.learning_rate,
        "per_device_train_batch_size": ft.train_batch_size,
        "per_device_eval_batch_size": ft.eval_batch_size,
        "num_train_epochs": ft.num_train_epochs,
        "weight_decay": ft.weight_decay,
        "warmup_ratio": ft.warmup_ratio,
        "gradient_accumulation_steps": ft.gradient_accumulation_steps,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": False,
        "report_to": "none",
        "seed": config.train.random_seed,
    }
    signature = inspect.signature(training_args_cls.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    return training_args_cls(**kwargs)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _json_number(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finetune a Transformer sequence classifier.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metrics = run_finetuning(config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

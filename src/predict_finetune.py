from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ProjectConfig, load_config
from src.data import LABEL_COLUMNS, load_test_frame
from src.submission import align_probabilities, make_submission_frame, write_submission


def run_finetune_prediction(
    config: ProjectConfig,
    output_path: str | Path,
    model_dir: str | Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    if config.finetune is None:
        raise ValueError("Missing finetune section in config.")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformer prediction requires optional dependencies. "
            "Install them with: pip install -r requirements-finetune.txt"
        ) from exc

    model_path = Path(model_dir) if model_dir is not None else config.finetune.output_dir
    if not model_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {model_path}. "
            "Train first with python -m src.finetune --config <config>."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_frame = load_test_frame(config.data.test_path, text_max_chars=config.data.text_max_chars)
    probabilities = _predict_probabilities(
        texts=test_frame["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        device=device,
        max_length=config.finetune.max_length,
        batch_size=batch_size or config.finetune.eval_batch_size,
    )
    class_order = class_order_from_model_config(model.config, probabilities.shape[1])
    probabilities = align_probabilities(probabilities, class_order)
    submission = make_submission_frame(test_frame["id"], probabilities)
    destination = write_submission(submission, output_path)

    return {
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "model_dir": str(model_path),
        "output_path": str(destination),
    }


def class_order_from_model_config(model_config: Any, num_labels: int) -> list[str]:
    id_to_label = getattr(model_config, "id2label", None) or {}
    class_order: list[str] = []
    for index in range(num_labels):
        label = id_to_label.get(index, id_to_label.get(str(index)))
        if label is None and index < len(LABEL_COLUMNS):
            label = LABEL_COLUMNS[index]
        class_order.append(str(label))
    return class_order


def _predict_probabilities(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    device: Any,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch_module.no_grad():
            logits = model(**encoded).logits
            batch_probabilities = torch_module.softmax(logits, dim=-1).detach().cpu().numpy()
        batches.append(batch_probabilities)

    if not batches:
        return np.empty((0, len(LABEL_COLUMNS)), dtype=float)
    return np.vstack(batches)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a submission from a fine-tuned Transformer model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    parser.add_argument("--model-dir", help="Fine-tuned model directory. Defaults to finetune.output_dir.")
    parser.add_argument("--batch-size", type=int, help="Prediction batch size. Defaults to finetune.eval_batch_size.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_finetune_prediction(
        config=config,
        output_path=args.output,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

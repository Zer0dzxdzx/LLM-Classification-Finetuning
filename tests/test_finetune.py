from __future__ import annotations

import unittest

import numpy as np

from src.config import load_config
from src.data import LABEL_COLUMNS
from src.finetune import TextClassificationDataset, _compute_metrics, _training_args
from src.predict_finetune import class_order_from_model_config
from tests.helpers import project_path


class FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = value

    def squeeze(self, axis: int) -> tuple[str, object, int]:
        return ("squeezed", self.value, axis)


class FakeTokenizer:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def __call__(self, text: str, **kwargs: object) -> dict[str, FakeTensor]:
        self.last_kwargs = kwargs
        return {
            "input_ids": FakeTensor([101, 102]),
            "attention_mask": FakeTensor([1, 1]),
        }


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(value: object, dtype: object) -> tuple[str, object, object]:
        return ("tensor", value, dtype)


class TrainingArgsWithEvalStrategy:
    def __init__(self, output_dir: str, eval_strategy: str, **kwargs: object) -> None:
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.kwargs = kwargs


class TrainingArgsWithEvaluationStrategy:
    def __init__(self, output_dir: str, evaluation_strategy: str, **kwargs: object) -> None:
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.kwargs = kwargs


class FakeModelConfig:
    id2label = {0: LABEL_COLUMNS[0], "1": LABEL_COLUMNS[1], 2: LABEL_COLUMNS[2]}


class FinetuneTests(unittest.TestCase):
    def test_dataset_encodes_text_and_label(self) -> None:
        tokenizer = FakeTokenizer()
        dataset = TextClassificationDataset(
            texts=["hello"],
            labels=[2],
            tokenizer=tokenizer,
            max_length=16,
            torch_module=FakeTorch,
        )

        item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(tokenizer.last_kwargs["max_length"], 16)
        self.assertEqual(tokenizer.last_kwargs["truncation"], True)
        self.assertEqual(item["input_ids"], ("squeezed", [101, 102], 0))
        self.assertEqual(item["labels"], ("tensor", 2, "long"))

    def test_compute_metrics_returns_log_loss_and_accuracy(self) -> None:
        logits = np.array([[4.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        labels = np.array([0, 1, 2])

        metrics = _compute_metrics((logits, labels))

        self.assertLess(metrics["log_loss"], 0.1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_training_args_supports_transformers_api_variants(self) -> None:
        config = load_config(project_path("configs", "finetune.yaml"))

        new_args = _training_args(config, TrainingArgsWithEvalStrategy)
        old_args = _training_args(config, TrainingArgsWithEvaluationStrategy)

        self.assertEqual(new_args.eval_strategy, "epoch")
        self.assertEqual(old_args.evaluation_strategy, "epoch")
        self.assertEqual(new_args.kwargs["report_to"], "none")
        self.assertEqual(old_args.kwargs["seed"], 42)

    def test_class_order_from_model_config_handles_int_and_string_keys(self) -> None:
        self.assertEqual(class_order_from_model_config(FakeModelConfig(), 3), LABEL_COLUMNS)


if __name__ == "__main__":
    unittest.main()

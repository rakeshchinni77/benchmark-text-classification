"""Fine-tuning support for DistilBERT on AG News."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import evaluate
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
	__version__ as transformers_version,
)

from src.config.config import (
	DEVICE,
	FINE_TUNE_MODEL,
	LABEL_NAMES,
	LEARNING_RATE,
	MAX_LENGTH,
	MODEL_OUTPUT_DIR,
	NUM_EPOCHS,
	TRAIN_BATCH_SIZE,
	EVAL_BATCH_SIZE,
)
from src.data.loader import load_ag_news
from src.evaluation.memory import get_peak_gpu_memory_mb

logger = logging.getLogger(__name__)


class FineTuner:
	"""Encapsulates supervised fine-tuning of DistilBERT for AG News."""

	def __init__(self, output_dir: Path | str | None = None, development_mode: bool = False) -> None:
		self.output_dir = Path(output_dir or MODEL_OUTPUT_DIR / "finetuned_full")
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.tokenizer = AutoTokenizer.from_pretrained(FINE_TUNE_MODEL)
		self.model = AutoModelForSequenceClassification.from_pretrained(
			FINE_TUNE_MODEL,
			num_labels=len(LABEL_NAMES),
		)
		self.development_mode = development_mode
		self.train_dataset, self.eval_dataset = self._prepare_datasets()
		self.metric_accuracy = evaluate.load("accuracy")
		self.metric_f1 = evaluate.load("f1")

	def _prepare_datasets(self) -> tuple[Any, Any]:
		"""Load AG News and tokenize the train and evaluation datasets."""

		dataset = load_ag_news()
		train_dataset = dataset["train"].rename_column("label", "labels")
		eval_dataset = dataset["test"].rename_column("label", "labels")

		if self.development_mode:
			train_dataset = train_dataset.select(range(min(5000, len(train_dataset))))
			eval_dataset = eval_dataset.select(range(min(1000, len(eval_dataset))))

		logger.info("Fine-tuning development mode=%s", self.development_mode)
		logger.info("Train dataset size=%d", len(train_dataset))
		logger.info("Eval dataset size=%d", len(eval_dataset))

		def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
			return self.tokenizer(
				batch["text"],
				padding="max_length",
				truncation=True,
				max_length=MAX_LENGTH,
			)

		train_dataset = train_dataset.map(
			tokenize_batch,
			batched=True,
			remove_columns=["text"],
		)
		eval_dataset = eval_dataset.map(
			tokenize_batch,
			batched=True,
			remove_columns=["text"],
		)

		# Use default format (not "torch") to avoid NumPy 2.x incompatibility
		# when datasets interacts with the Trainer's compute_metrics callback.
		# The Trainer will handle tensor conversion automatically.
		logger.info("Dataset preparation complete: using default format for Trainer compatibility")

		return train_dataset, eval_dataset

	def _reset_gpu_peak_memory(self) -> None:
		if not torch.cuda.is_available():
			return

		try:
			torch.cuda.reset_peak_memory_stats()
		except Exception:
			logger.exception("Unable to reset GPU memory statistics")

	def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
		predictions, labels = eval_pred
		# argmax is available on both numpy arrays and torch tensors
		predictions_idx = predictions.argmax(axis=-1)
		accuracy_result = self.metric_accuracy.compute(predictions=predictions_idx, references=labels)
		f1_result = self.metric_f1.compute(predictions=predictions_idx, references=labels, average="macro")
		return {
			"accuracy": float(accuracy_result["accuracy"]),
			"macro_f1": float(f1_result["f1"]),
		}

	def train(self) -> dict[str, Any]:
		"""Train DistilBERT and return training and evaluation results."""

		training_args = TrainingArguments(
			output_dir=str(self.output_dir),
			num_train_epochs=NUM_EPOCHS,
			per_device_train_batch_size=TRAIN_BATCH_SIZE,
			per_device_eval_batch_size=EVAL_BATCH_SIZE,
			evaluation_strategy="epoch",
			save_strategy="epoch",
			save_total_limit=1,
			load_best_model_at_end=True,
			metric_for_best_model="accuracy",
			learning_rate=LEARNING_RATE,
			fp16=(DEVICE == "cuda"),
			report_to="none",
		)

		logger.info("Fine-tuning model=%s", FINE_TUNE_MODEL)
		logger.info("Transformers version=%s", transformers_version)
		logger.info("fp16 enabled=%s", DEVICE == "cuda")

		data_collator = DataCollatorWithPadding(self.tokenizer)

		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=self.train_dataset,
			eval_dataset=self.eval_dataset,
			tokenizer=self.tokenizer,
			data_collator=data_collator,
			compute_metrics=self._compute_metrics,
		)

		self._reset_gpu_peak_memory()
		start_time = time.perf_counter()
		trainer.train()
		training_time_minutes = (time.perf_counter() - start_time) / 60.0

		trainer.save_model(str(self.output_dir))
		self.tokenizer.save_pretrained(str(self.output_dir))

		peak_training_gpu_memory_mb = get_peak_gpu_memory_mb()
		evaluation_metrics = trainer.evaluate()

		return {
			"training_time_minutes": float(training_time_minutes),
			"peak_training_gpu_memory_mb": float(peak_training_gpu_memory_mb),
			"evaluation": {
				"accuracy": float(evaluation_metrics["eval_accuracy"]),
				"macro_f1": float(evaluation_metrics["eval_macro_f1"]),
			},
		}


__all__ = ["FineTuner"]

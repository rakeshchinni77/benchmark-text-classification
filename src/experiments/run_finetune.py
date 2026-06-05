"""Fine-tuning experiment runner for DistilBERT on AG News."""

from __future__ import annotations

from typing import Any

from src.models.fine_tune import FineTuner


def run_finetune_experiment(development_mode: bool = False) -> dict[str, Any]:
	"""Fine-tune DistilBERT and return training metrics."""

	tuner = FineTuner(development_mode=development_mode)
	return tuner.train()


def run_finetune_smoke_test() -> dict[str, Any]:
	"""Run a lightweight fine-tuning smoke test using the development subset."""

	return run_finetune_experiment(development_mode=True)


__all__ = ["run_finetune_experiment", "run_finetune_smoke_test"]

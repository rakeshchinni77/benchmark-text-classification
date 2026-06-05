"""Few-shot experiment runner."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from src.data.loader import get_test_dataset
from src.evaluation.evaluator import run_evaluation
from src.evaluation.memory import get_peak_gpu_memory_mb
from src.models.few_shot import FewShotClassifier

logger = logging.getLogger(__name__)


SUPPORTED_FEW_SHOT_K_VALUES = [2, 4, 8, 16]


def _reset_gpu_peak_memory() -> None:
	"""Reset CUDA peak memory statistics if a GPU is available."""

	try:
		import torch
	except ImportError:
		return

	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()


def run_few_shot_experiment(development: bool = True) -> dict[str, Any]:
	"""Run the few-shot AG News benchmark and return results grouped by k."""

	mode_name = "development" if development else "final"
	sample_size = 50 if development else 500

	logger.info("Running few-shot experiment in %s mode", mode_name)
	logger.info("Requesting %d test examples", sample_size)

	dataset = get_test_dataset()
	selected_size = min(sample_size, len(dataset))
	subset = dataset.select(range(selected_size))
	classifier = FewShotClassifier()

	_reset_gpu_peak_memory()

	results: dict[str, Any] = {}
	for k in SUPPORTED_FEW_SHOT_K_VALUES:
		logger.info("Evaluating few-shot setting k=%d", k)
		predict_fn = partial(classifier.predict, k=k)
		results[f"k_{k}"] = run_evaluation(predict_fn, subset)

	results["peak_gpu_memory_mb"] = get_peak_gpu_memory_mb()
	logger.info("Few-shot experiment completed")

	return results


__all__ = ["run_few_shot_experiment"]

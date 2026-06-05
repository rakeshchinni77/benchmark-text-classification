"""Zero-shot experiment runner."""

from __future__ import annotations

import logging
from typing import Any

from src.config.config import FULL_TEST_SIZE
from src.data.loader import get_test_dataset
from src.evaluation.evaluator import run_evaluation
from src.evaluation.memory import get_peak_gpu_memory_mb
from src.models.zero_shot import ZeroShotClassifier

logger = logging.getLogger(__name__)


def _reset_gpu_peak_memory() -> None:
	"""Reset CUDA peak memory statistics if a GPU is available."""

	try:
		import torch
	except ImportError:
		return

	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()


def run_zero_shot_experiment(development: bool = True) -> dict[str, float]:
	"""Run the zero-shot AG News benchmark and return evaluation metrics."""

	mode_name = "development" if development else "final"
	sample_size = 100 if development else FULL_TEST_SIZE

	dataset = get_test_dataset()
	logger.info(
		"Running zero-shot experiment in %s mode with %d examples",
		mode_name,
		sample_size,
	)

	selected_size = min(sample_size, len(dataset))
	subset = dataset.select(range(selected_size))

	classifier = ZeroShotClassifier()
	_reset_gpu_peak_memory()

	results = run_evaluation(classifier.predict, subset)
	peak_gpu_memory_mb = get_peak_gpu_memory_mb()

	metrics = {
		"accuracy": float(results["accuracy"]),
		"macro_f1": float(results["macro_f1"]),
		"median_latency_ms": float(results["median_latency_ms"]),
		"p99_latency_ms": float(results["p99_latency_ms"]),
		"peak_gpu_memory_mb": float(peak_gpu_memory_mb),
	}

	logger.info(
		"Zero-shot experiment completed: %s",
		metrics,
	)

	return metrics


__all__ = ["run_zero_shot_experiment"]

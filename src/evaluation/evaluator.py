"""Universal evaluation orchestration for benchmark prediction functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from src.evaluation.latency import compute_latency_statistics, measure_latency
from src.evaluation.metrics import compute_accuracy, compute_macro_f1

logger = logging.getLogger(__name__)


def run_evaluation(
	predict_fn: Callable[[str], int],
	dataset: Iterable[dict[str, Any]],
) -> dict[str, float]:
	"""Evaluate a prediction function over a dataset.

	The function expects each dataset item to provide ``text`` and ``label``
	fields. Invalid predictions are skipped safely and do not stop evaluation.
	"""

	predictions: list[int] = []
	references: list[int] = []
	latencies_ms: list[float] = []

	for example in dataset:
		text = example.get("text", "")
		label = example.get("label")

		if label is None:
			logger.warning("Skipping example without a label")
			continue

		try:
			predicted_label, latency_ms = measure_latency(lambda: predict_fn(text))
		except Exception:
			logger.exception("Prediction function failed; skipping example")
			continue

		latencies_ms.append(latency_ms)

		if not isinstance(predicted_label, int) or predicted_label < 0:
			logger.debug("Skipping invalid prediction: %s", predicted_label)
			continue

		predictions.append(predicted_label)
		references.append(int(label))

	if not predictions:
		latency_stats = compute_latency_statistics(latencies_ms)
		return {
			"accuracy": 0.0,
			"macro_f1": 0.0,
			"median_latency_ms": latency_stats["median_latency_ms"],
			"p99_latency_ms": latency_stats["p99_latency_ms"],
		}

	latency_stats = compute_latency_statistics(latencies_ms)

	return {
		"accuracy": compute_accuracy(predictions, references),
		"macro_f1": compute_macro_f1(predictions, references),
		"median_latency_ms": latency_stats["median_latency_ms"],
		"p99_latency_ms": latency_stats["p99_latency_ms"],
	}


__all__ = ["run_evaluation"]

"""Metric helpers for the benchmark evaluation framework.

The functions in this module are intentionally small and reusable so future
benchmark phases can compute classification metrics without duplicating logic.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

try:
	import evaluate
except ImportError:  # pragma: no cover - optional dependency in some envs
	evaluate = None


@lru_cache(maxsize=1)
def _accuracy_metric():
	"""Load and cache the Hugging Face accuracy metric."""

	if evaluate is None:
		raise ImportError("The 'evaluate' package is required to compute accuracy.")
	return evaluate.load("accuracy")


@lru_cache(maxsize=1)
def _f1_metric():
	"""Load and cache the Hugging Face macro F1 metric."""

	if evaluate is None:
		raise ImportError("The 'evaluate' package is required to compute macro F1.")
	return evaluate.load("f1")


def compute_accuracy(predictions: Sequence[int], references: Sequence[int]) -> float:
	"""Compute classification accuracy as a float."""

	metric = _accuracy_metric()
	result = metric.compute(predictions=list(predictions), references=list(references))
	return float(result["accuracy"])


def compute_macro_f1(predictions: Sequence[int], references: Sequence[int]) -> float:
	"""Compute macro-averaged F1 score as a float."""

	metric = _f1_metric()
	result = metric.compute(
		predictions=list(predictions),
		references=list(references),
		average="macro",
	)
	return float(result["f1"])


__all__ = ["compute_accuracy", "compute_macro_f1"]

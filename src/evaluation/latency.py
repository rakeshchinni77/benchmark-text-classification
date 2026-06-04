"""Latency measurement utilities for benchmark predictions."""

from __future__ import annotations

import logging
import statistics
import time
from typing import Callable, Iterable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def measure_latency(function: Callable[[], T]) -> tuple[T, float]:
	"""Execute ``function`` once and return its result with latency in ms."""

	start_time = time.perf_counter()
	result = function()
	end_time = time.perf_counter()
	latency_ms = (end_time - start_time) * 1000.0
	return result, latency_ms


def compute_latency_statistics(latencies_ms: Iterable[float]) -> dict[str, float]:
	"""Return median and p99 latency statistics in milliseconds."""

	latency_values = list(latencies_ms)
	if not latency_values:
		return {"median_latency_ms": 0.0, "p99_latency_ms": 0.0}

	sorted_latencies = sorted(latency_values)
	median_latency_ms = float(statistics.median(sorted_latencies))

	percentile_index = max(0, min(len(sorted_latencies) - 1, int(round(0.99 * (len(sorted_latencies) - 1)))))
	p99_latency_ms = float(sorted_latencies[percentile_index])

	return {
		"median_latency_ms": median_latency_ms,
		"p99_latency_ms": p99_latency_ms,
	}


__all__ = ["compute_latency_statistics", "measure_latency"]

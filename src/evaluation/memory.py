"""GPU memory tracking helpers for benchmark evaluation.

The functions in this module are safe on CPU-only systems. If CUDA is not
available, memory reporting falls back to ``0.0`` instead of raising errors.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
	import torch
except ImportError:  # pragma: no cover - optional dependency in some envs
	torch = None


def get_peak_gpu_memory_mb() -> float:
	"""Return the peak CUDA memory usage in megabytes.

	On CPU-only systems, or when PyTorch is unavailable, the function returns
	``0.0`` and does not fail.
	"""

	if torch is None:
		return 0.0

	try:
		if not torch.cuda.is_available():
			return 0.0
		return float(torch.cuda.max_memory_allocated() / (1024**2))
	except Exception:
		logger.exception("Unable to read peak GPU memory")
		return 0.0


__all__ = ["get_peak_gpu_memory_mb"]

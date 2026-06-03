"""Text preprocessing helpers for AG News data preparation.

These functions intentionally stay minimal: they normalize whitespace and
prepare raw text for downstream benchmark phases without tokenizing or
performing any model-specific processing.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
	"""Normalize a single text value by trimming and collapsing whitespace."""

	normalized_text = text.strip()
	normalized_text = _WHITESPACE_PATTERN.sub(" ", normalized_text)
	return normalized_text


def preprocess_batch(texts: Iterable[str]) -> list[str]:
	"""Normalize a collection of text values using ``preprocess_text``."""

	logger.debug("Preprocessing text batch")
	return [preprocess_text(text) for text in texts]


__all__ = ["preprocess_batch", "preprocess_text"]

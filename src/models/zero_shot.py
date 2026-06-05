"""Zero-shot classification model support."""

from __future__ import annotations

import logging
from typing import Any

from src.config.config import DEVICE, LABEL_NAMES, ZERO_SHOT_MODEL

logger = logging.getLogger(__name__)


class ZeroShotClassifier:
	"""Zero-shot classifier using a Hugging Face pipeline."""

	CANDIDATE_LABELS: dict[int, str] = {
		0: "World news, politics, international affairs",
		1: "Sports news, games, athletes, competitions",
		2: "Business, finance, economy, markets",
		3: "Science, technology, research, innovation",
	}

	def __init__(self) -> None:
		self.device = 0 if DEVICE == "cuda" else -1
		self.candidate_labels = list(self.CANDIDATE_LABELS.values())
		self.classifier = self._load_pipeline()

	def _load_pipeline(self) -> Any:
		"""Load the zero-shot classification pipeline."""

		try:
			from transformers import pipeline
		except ImportError as exc:
			logger.exception("Transformers dependency is missing")
			raise ImportError(
				"The transformers package is required for ZeroShotClassifier. "
				"Install it with 'pip install transformers torch'"
			) from exc

		logger.info(
			"Loading zero-shot model '%s' on device %s",
			ZERO_SHOT_MODEL,
			self.device,
		)
		return pipeline(
			task="zero-shot-classification",
			model=ZERO_SHOT_MODEL,
			tokenizer=ZERO_SHOT_MODEL,
			device=self.device,
		)

	def predict(self, text: str) -> int:
		"""Predict the label index for a single input text."""

		if not isinstance(text, str):
			raise TypeError("text must be a string")

		result = self.classifier(text, candidate_labels=self.candidate_labels)
		if not isinstance(result, dict):
			logger.error("Unexpected zero-shot output format: %s", result)
			raise ValueError("Zero-shot classifier returned an invalid result")

		labels = result.get("labels")
		if not labels:
			logger.error("Zero-shot classifier returned no labels for text: %s", text)
			raise ValueError("Zero-shot classifier did not return any labels")

		predicted_label = labels[0]
		for label_id, label_text in self.CANDIDATE_LABELS.items():
			if label_text == predicted_label:
				return label_id

		logger.exception("Predicted label is not in candidate hypotheses: %s", predicted_label)
		raise ValueError("Predicted label is not a configured candidate hypothesis")


__all__ = ["ZeroShotClassifier"]

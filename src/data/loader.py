"""Dataset loading and validation utilities for the AG News benchmark.

This module is intentionally limited to data access, label mapping, dataset
inspection, and validation helpers. It does not perform any tokenization,
model loading, or inference work so future phases can reuse it safely.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from datasets import DatasetDict, load_dataset

from src.config.config import DATASET_NAME, LABEL_NAMES

logger = logging.getLogger(__name__)

LABEL_MAPPING: dict[int, str] = {
	0: "World",
	1: "Sports",
	2: "Business",
	3: "Science and Technology",
}
"""Canonical mapping from label ids to human-readable names."""

ID_TO_LABEL: dict[int, str] = LABEL_MAPPING.copy()
"""Alias for the id-to-label mapping used by later phases."""

LABEL_TO_ID: dict[str, int] = {label: label_id for label_id, label in LABEL_MAPPING.items()}
"""Inverse mapping from human-readable label names to label ids."""


@lru_cache(maxsize=1)
def load_ag_news() -> DatasetDict:
	"""Load the AG News dataset from the Hugging Face datasets hub.

	Returns:
		A cached ``DatasetDict`` containing the ``train`` and ``test`` splits.
	"""

	logger.info("Loading dataset '%s'", DATASET_NAME)
	return load_dataset(DATASET_NAME)


def get_train_dataset():
	"""Return the AG News training split."""

	dataset = load_ag_news()
	return dataset["train"]


def get_test_dataset():
	"""Return the AG News test split."""

	dataset = load_ag_news()
	return dataset["test"]


def get_dataset_info() -> dict[str, Any]:
	"""Return basic dataset statistics for reporting and validation."""

	train_dataset = get_train_dataset()
	test_dataset = get_test_dataset()

	return {
		"train_size": len(train_dataset),
		"test_size": len(test_dataset),
		"label_names": LABEL_NAMES,
		"num_classes": len(LABEL_NAMES),
	}


def validate_dataset() -> bool:
	"""Validate that the AG News dataset is present and structurally sound.

	Checks that the expected splits exist, the dataset is non-empty, and all
	observed labels are valid according to the centralized label mapping.
	"""

	dataset = load_ag_news()

	if "train" not in dataset or "test" not in dataset:
		logger.error("Dataset is missing required train/test splits")
		return False

	train_dataset = dataset["train"]
	test_dataset = dataset["test"]

	if len(train_dataset) == 0 or len(test_dataset) == 0:
		logger.error("Dataset splits must not be empty")
		return False

	valid_label_ids = set(ID_TO_LABEL.keys())
	observed_labels = set(train_dataset["label"]) | set(test_dataset["label"])

	if not observed_labels.issubset(valid_label_ids):
		logger.error("Dataset contains invalid label ids: %s", observed_labels - valid_label_ids)
		return False

	logger.info("Dataset validation passed")
	return True


__all__ = [
	"ID_TO_LABEL",
	"LABEL_MAPPING",
	"LABEL_TO_ID",
	"get_dataset_info",
	"get_test_dataset",
	"get_train_dataset",
	"load_ag_news",
	"validate_dataset",
]

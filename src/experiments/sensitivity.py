"""Few-shot sensitivity experiment runner."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np

from src.config.config import LABEL_NAMES
from src.data.loader import get_test_dataset, get_train_dataset
from src.evaluation.evaluator import run_evaluation
from src.models.few_shot import FewShotClassifier

logger = logging.getLogger(__name__)


def run_sensitivity_experiment() -> dict[str, Any]:
	"""Run few-shot sensitivity experiments for k=8 using multiple seeds."""

	seeds = [42, 123, 999]
	k = 8

	dataset = get_test_dataset()
	development_size = min(50, len(dataset))
	subset = dataset.select(range(development_size))
	train_dataset = get_train_dataset()

	classifier = FewShotClassifier()
	seed_results: dict[str, float] = {}
	accuracies: list[float] = []

	for seed in seeds:
		logger.info("Running sensitivity seed=%d", seed)
		rng = np.random.default_rng(seed)
		sample_indices = rng.choice(len(train_dataset), size=k, replace=False)
		selected_examples: list[dict[str, str]] = []

		for index in sample_indices:
			example = train_dataset[int(index)]
			label_id = int(example.get("label", -1))
			label_text = LABEL_NAMES[label_id] if 0 <= label_id < len(LABEL_NAMES) else "Unknown"
			selected_examples.append({"text": example.get("text", ""), "label": label_text})

		classifier.examples_by_k[k] = selected_examples

		predict_fn = partial(classifier.predict, k=k)
		metrics = run_evaluation(predict_fn, subset)

		accuracy = float(metrics["accuracy"])
		seed_results[str(seed)] = accuracy
		accuracies.append(accuracy)

	accuracy_mean = float(np.mean(accuracies))
	accuracy_std = float(np.std(accuracies))

	return {
		"seed_results": seed_results,
		"accuracy_mean": accuracy_mean,
		"accuracy_std": accuracy_std,
	}


__all__ = ["run_sensitivity_experiment"]

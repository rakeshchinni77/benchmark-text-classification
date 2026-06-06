"""Data efficiency experiment for DistilBERT on AG News.

This module trains DistilBERT on progressively larger subsets of the AG News
training data to measure how much labeled data is needed to achieve strong
performance. Results are saved to results/data_efficiency.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config.config import MODEL_OUTPUT_DIR, RESULTS_DIR, TRAIN_SUBSETS
from src.models.fine_tune import FineTuner

logger = logging.getLogger(__name__)


def run_data_efficiency_experiment(development_mode: bool = False) -> dict[str, Any]:
	"""Train DistilBERT on different training subset sizes and collect accuracy.

	For each subset size, trains a fresh model and evaluates on the full AG News
	test set. Results are collected and saved to results/data_efficiency.json.

	Args:
		development_mode: If True, only train on subsets [100, 500]. If False,
			train on all subsets [100, 500, 2000, 10000, 120000].

	Returns:
		A dict with accuracy results for each subset size:
		{
			"100": {"accuracy": float},
			"500": {"accuracy": float},
			"2000": {"accuracy": float},
			"10000": {"accuracy": float},
			"120000": {"accuracy": float},
		}
		(or only 100 and 500 if development_mode=True)
	"""

	subset_sizes = [100, 500] if development_mode else TRAIN_SUBSETS

	logger.info("Starting data efficiency experiment")
	logger.info("Development mode: %s", development_mode)
	logger.info("Subset sizes: %s", subset_sizes)

	results = {}

	for train_size in subset_sizes:
		logger.info("=" * 70)
		logger.info("Training with subset size: %d", train_size)
		logger.info("=" * 70)

		# Set output directory for this subset
		output_dir = MODEL_OUTPUT_DIR / f"finetuned_{train_size}"

		# Create FineTuner with custom train_size
		finetuner = FineTuner(
			output_dir=output_dir,
			development_mode=False,
			train_size=train_size,
		)

		# Train and collect results
		training_results = finetuner.train()

		# Extract accuracy only
		accuracy = training_results["evaluation"]["accuracy"]
		results[str(train_size)] = {"accuracy": accuracy}

		logger.info("Subset size %d: accuracy=%.4f", train_size, accuracy)

	# Save results to JSON
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	results_path = RESULTS_DIR / "data_efficiency.json"
	with open(results_path, "w") as f:
		json.dump(results, f, indent=2)
	logger.info("Results saved to %s", results_path)

	return results


__all__ = ["run_data_efficiency_experiment"]

"""Aggregate precomputed experiment results into `results/results.json`.

This module reads previously generated JSON files from the `results/`
directory and merges them into a single output document. No model
training or benchmark execution is performed by this aggregator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config.config import RESULTS_DIR, RESULTS_JSON_PATH

logger = logging.getLogger(__name__)

RESULT_FILES = {
    "zero_shot": "zero_shot.json",
    "few_shot": "few_shot.json",
    "sensitivity": "sensitivity.json",
    "fine_tuning": "fine_tuning.json",
    "inference": "inference.json",
    "data_efficiency": "data_efficiency.json",
}


def _load_json(filename: str) -> dict[str, Any]:
    """Load a JSON file from the results directory and return its contents.

    Returns an empty dict if the file is missing or cannot be parsed.
    """
    path = RESULTS_DIR / filename
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.warning("Missing results file: %s", path)
        return {}
    except Exception:
        logger.exception("Failed to load JSON from %s", path)
        return {}


def generate_results_json() -> dict[str, Any]:
    """Aggregate precomputed results and write `results/results.json`."""

    logger.info("Aggregating precomputed results into %s", RESULTS_JSON_PATH)

    aggregated: dict[str, Any] = {}
    for key, filename in RESULT_FILES.items():
        aggregated[key] = _load_json(filename)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as fh:
            json.dump(aggregated, fh, indent=2)
        logger.info("Wrote aggregated results to %s", RESULTS_JSON_PATH)
    except Exception:
        logger.exception("Failed to write aggregated results to %s", RESULTS_JSON_PATH)

    return aggregated


__all__ = ["generate_results_json"]

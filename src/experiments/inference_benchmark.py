"""Inference benchmarking for fine-tuned DistilBERT on AG News.

This module measures latency of the fine-tuned model saved at outputs/models/finetuned_full/
using two batch sizes (1 and 128) on AG News test examples.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.config import MODEL_OUTPUT_DIR
from src.data.loader import get_test_dataset
from src.evaluation.latency import compute_latency_statistics, measure_latency

BENCHMARK_SAMPLE_SIZE = 256

logger = logging.getLogger(__name__)


def run_inference_benchmark() -> dict[str, Any]:
	"""Benchmark inference latency on the fine-tuned DistilBERT model.

	Loads the fine-tuned model from outputs/models/finetuned_full/ and measures
	inference latency with batch_size=1 and batch_size=128 on AG News test samples.

	Returns:
		A dict with latency statistics for each batch size and throughput comparison:
		{
			"batch_size_1": {
				"median_latency_ms": float,
				"p99_latency_ms": float,
				"per_sample_latency_ms": float,
			},
			"batch_size_128": {
				"median_latency_ms": float,
				"p99_latency_ms": float,
				"per_sample_latency_ms": float,
			},
			"throughput_improved": bool,
		}
	"""

	model_dir = MODEL_OUTPUT_DIR / "finetuned_full"
	logger.info("Loading fine-tuned model from %s", model_dir)

	# Load model and tokenizer
	tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
	model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
	model.eval()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = model.to(device)
	logger.info("Model loaded on device: %s", device)

	# Load benchmark dataset subset
	test_dataset = get_test_dataset().select(range(BENCHMARK_SAMPLE_SIZE))
	logger.info("Benchmark dataset size: %d", len(test_dataset))

	# Benchmark with different batch sizes
	results = {}
	for batch_size in [1, 128]:
		num_batches = (len(test_dataset) + batch_size - 1) // batch_size
		estimated_runtime_minutes = num_batches * 1.0 / 60.0
		logger.info("Benchmarking with batch_size=%d", batch_size)
		logger.info("Number of batches for batch_size=%d: %d", batch_size, num_batches)
		logger.info(
			"Estimated runtime for batch_size=%d: %.2f minutes",
			batch_size,
			estimated_runtime_minutes,
		)
		latencies_ms = _benchmark_batch_size(
			model=model,
			tokenizer=tokenizer,
			dataset=test_dataset,
			batch_size=batch_size,
			device=device,
		)

		stats = compute_latency_statistics(latencies_ms)
		per_sample_latency_ms = stats["median_latency_ms"] / batch_size
		stats["per_sample_latency_ms"] = per_sample_latency_ms
		results[f"batch_size_{batch_size}"] = stats
		logger.info(
			"  batch_size=%d: median_latency_ms=%.2f, p99_latency_ms=%.2f, per_sample_latency_ms=%.2f",
			batch_size,
			stats["median_latency_ms"],
			stats["p99_latency_ms"],
			per_sample_latency_ms,
		)

	# Compute throughput improvement verification
	batch_size_1_per_sample = results["batch_size_1"]["per_sample_latency_ms"]
	batch_size_128_per_sample = results["batch_size_128"]["per_sample_latency_ms"]
	throughput_improved = batch_size_128_per_sample < batch_size_1_per_sample
	results["throughput_improved"] = throughput_improved
	logger.info(
		"Throughput improved with larger batch size: %s (batch_size_128: %.2f ms vs batch_size_1: %.2f ms per sample)",
		throughput_improved,
		batch_size_128_per_sample,
		batch_size_1_per_sample,
	)

	return results


def _benchmark_batch_size(
	model: Any,
	tokenizer: Any,
	dataset: Any,
	batch_size: int,
	device: str,
) -> list[float]:
	"""Run inference benchmarking for a specific batch size.

	Args:
		model: Fine-tuned DistilBERT model in eval mode.
		tokenizer: Tokenizer for the model.
		dataset: AG News test dataset with 'text' field.
		batch_size: Batch size for inference.
		device: Device to run on ("cuda" or "cpu").

	Returns:
		List of latencies in milliseconds.
	"""

	latencies_ms = []

	# Process dataset in batches
	num_batches = (len(dataset) + batch_size - 1) // batch_size
	for batch_idx in range(num_batches):
		start_idx = batch_idx * batch_size
		end_idx = min(start_idx + batch_size, len(dataset))

		# Extract batch texts
		batch_texts = [dataset[i]["text"] for i in range(start_idx, end_idx)]

		# Measure tokenization + inference latency
		def run_inference():
			# Tokenize
			inputs = tokenizer(
				batch_texts,
				padding="max_length",
				truncation=True,
				max_length=512,
				return_tensors="pt",
			)
			inputs = {k: v.to(device) for k, v in inputs.items()}

			# Forward pass
			with torch.no_grad():
				outputs = model(**inputs)

			return outputs

		_, latency_ms = measure_latency(run_inference)
		latencies_ms.append(latency_ms)

	return latencies_ms


__all__ = ["run_inference_benchmark"]

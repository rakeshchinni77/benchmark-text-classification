"""Few-shot classification model support."""

from __future__ import annotations

import logging
from typing import Any

try:
	import torch
except ImportError as exc:  # pragma: no cover - optional dependency
	torch = None
	raise ImportError("The torch package is required for few-shot inference") from exc

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config.config import DEVICE, FEW_SHOT_MODEL, LABEL_NAMES
from src.data.loader import get_train_dataset

logger = logging.getLogger(__name__)


class FewShotClassifier:
	"""Few-shot classifier using a FLAN-T5 text generation model."""

	SUPPORTED_K_VALUES = [2, 4, 8, 16]
	EXAMPLE_TEXT_MAX_LENGTH = 120
	MAX_TOKEN_LENGTH = 512

	def __init__(self) -> None:
		self.device = torch.device("cuda" if DEVICE == "cuda" else "cpu")
		self.examples_by_k = self._build_fixed_examples()
		self.tokenizer = AutoTokenizer.from_pretrained(FEW_SHOT_MODEL)
		# Respect model's maximum encoder length but cap at 512 for FLAN-T5-base
		self.max_model_length = min(getattr(self.tokenizer, "model_max_length", 512) or 512, 512)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(FEW_SHOT_MODEL)
		self.model.to(self.device)
		self.model.eval()

	def _build_fixed_examples(self) -> dict[int, list[dict[str, str]]]:
		"""Build deterministic few-shot example collections from the training split."""

		train_dataset = get_train_dataset()
		all_examples: list[dict[str, str]] = []
		for index in range(max(self.SUPPORTED_K_VALUES)):
			example = train_dataset[index]
			label_id = int(example.get("label", -1))
			label_text = LABEL_NAMES[label_id] if 0 <= label_id < len(LABEL_NAMES) else "Unknown"
			all_examples.append({"text": example.get("text", ""), "label": label_text})

		return {k: all_examples[:k] for k in self.SUPPORTED_K_VALUES}

	def get_examples(self, k: int) -> list[dict[str, str]]:
		"""Return the fixed examples used for a specific few-shot setting."""

		if k not in self.SUPPORTED_K_VALUES:
			raise ValueError(f"Unsupported few-shot value: {k}")
		return self.examples_by_k[k]

	def _shorten_example_text(self, text: str, max_length: int) -> str:
		"""Shorten example text to a fixed length while preserving readability."""

		if len(text) <= max_length:
			return text
		return text[:max_length].rstrip() + "..."

	def _build_prompt(self, query_text: str, examples: list[dict[str, str]], max_example_chars: int) -> str:
		"""Build a few-shot prompt from the provided examples and query text."""

		example_lines = []
		for example in examples:
			text = self._shorten_example_text(example.get("text", ""), max_example_chars)
			label = example.get("label", "")
			example_lines.append(f"Text: {text}\nCategory: {label}")

		return (
			"Classify the following news article into one of:\n\n"
			"World\n"
			"Sports\n"
			"Business\n"
			"Science and Technology\n\n"
			"Examples:\n\n"
			+ "\n\n".join(example_lines)
			+ "\n\n"
			+ f"Text: {query_text}\nCategory:"
		)

	def build_prompt(self, query_text: str, examples: list[dict[str, str]]) -> str:
		"""Build a prompt and ensure it fits within the maximum token budget.

		This implementation constructs the prompt incrementally. It starts with
		the instruction block and the label list, then appends examples one at
		a time. The query text is preserved as the final block, and the function
		stops adding examples once the model encoder limit would be exceeded.
		The returned prompt always preserves the instruction, the query, and the
		final "Category:" completion slot.
		"""

		# Base components: instruction, labels, and the query text (preserved)
		instruction_and_labels = (
			"Classify the following news article into one of:\n\n"
			"World\n"
			"Sports\n"
			"Business\n"
			"Science and Technology\n\n"
		)

		query_block = f"Text: {query_text}\nCategory:"  # always preserved

		examples_header = "\n\nExamples:\n\n"

		prompt_prefix = instruction_and_labels + examples_header

		# Start with instruction, label list, and examples section. Examples are added
		# incrementally; the query block is only appended for token counting.
		current_prompt = prompt_prefix

		for example in examples:
			short_text = self._shorten_example_text(example.get("text", ""), self.EXAMPLE_TEXT_MAX_LENGTH)
			example_line = f"Text: {short_text}\nCategory: {example.get('label', '')}\n\n"

			# Candidate prompt if we add this example and then append the query block.
			candidate_prompt = current_prompt + example_line + query_block

			# Compute raw token count (no truncation) to decide whether to include
			candidate_count = self.get_prompt_token_count(candidate_prompt)

			if candidate_count > self.max_model_length:
				logger.debug("Stopping example addition: next example would exceed token limit (%d>%d)", candidate_count, self.max_model_length)
				break

			# Accept the example and keep the query block separate.
			current_prompt += example_line

		# Final safety: ensure we do not exceed the model length with the query
		final_count = self.get_prompt_token_count(current_prompt + query_block)
		logger.info("Final few-shot prompt token count=%d (limit=%d)", final_count, self.max_model_length)

		# As a last resort, if still too long, truncate example texts aggressively
		if final_count > self.max_model_length:
			logger.warning("Prompt still exceeds model length after example selection; truncating examples further")
			# Remove examples until within limit
			parts = current_prompt.split("\n\nExamples:\n\n")
			if len(parts) == 2:
				prefix, examples_blob = parts
				ex_lines = examples_blob.strip().split("\n\n")
				while ex_lines and self.get_prompt_token_count(prefix + "\n\nExamples:\n\n" + "\n\n".join(ex_lines) + query_block) > self.max_model_length:
					ex_lines.pop()
				current_prompt = prefix + "\n\nExamples:\n\n" + "\n\n".join(ex_lines)

		return current_prompt + query_block

	def get_prompt_token_count(self, prompt: str) -> int:
		"""Return the number of tokens the tokenizer produces for `prompt`.

		This counts tokens without truncation so we can decide whether adding an
		example would overflow the model encoder. The tokenizer is called in
		raw mode to get the full token count.
		"""

		# The tokenizer emits a warning when the produced token sequence
		# exceeds `tokenizer.model_max_length`. We need the raw token count
		# (without truncation) to decide whether adding an example would
		# overflow the model encoder, but we must avoid emitting the
		# warning. To safely compute the raw token count we temporarily
		# raise `model_max_length` to a very large value, call the
		# tokenizer, then restore the original value.
		old_max = getattr(self.tokenizer, "model_max_length", None)
		try:
			# Use a very large sentinel to avoid warnings while counting
			self.tokenizer.model_max_length = 10 ** 12
			count = len(self.tokenizer(prompt, add_special_tokens=True).input_ids)
		finally:
			# Restore prior model_max_length (fall back to self.max_model_length)
			if old_max is None:
				self.tokenizer.model_max_length = self.max_model_length
			else:
				self.tokenizer.model_max_length = old_max

		return count

	def _parse_output(self, output: str) -> int:
		"""Convert generated text into a numeric AG News label index."""

		if not isinstance(output, str):
			return -1

		normalized = output.strip().lower()
		if not normalized:
			return -1

		# Prefer direct numeric answers if the model returns them.
		if normalized.isdigit():
			candidate = int(normalized)
			if 0 <= candidate < len(LABEL_NAMES):
				return candidate

		# Normalize common text output forms.
		normalized = normalized.strip(".:- ")
		for index, label in enumerate(LABEL_NAMES):
			label_lower = label.lower()
			if normalized == label_lower:
				return index
			if normalized.startswith(label_lower):
				return index
			if label_lower in normalized:
				return index

		return -1

	def predict(self, text: str, k: int) -> int:
		"""Predict the label index for a single text using k-shot examples."""

		if not isinstance(text, str):
			raise TypeError("text must be a string")
		if k not in self.SUPPORTED_K_VALUES:
			raise ValueError(f"Unsupported few-shot value: {k}")

		examples = self.examples_by_k.get(k)
		if examples is None:
			raise ValueError(f"No examples available for k={k}")

		prompt = self.build_prompt(text, examples)
		token_count = len(
			self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_model_length).input_ids
		)
		logger.info("Few-shot prompt token count=%d for k=%d", token_count, k)

		inputs = self.tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=self.max_model_length,
		)
		inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

		output_ids = self.model.generate(
			**inputs,
			max_new_tokens=16,
			num_beams=1,
		)
		decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
		return self._parse_output(decoded)


__all__ = ["FewShotClassifier"]

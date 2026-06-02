"""Phase 1 entrypoint for container startup validation."""

from pathlib import Path


def main() -> None:
	"""Print the bootstrap message and create the expected runtime folders."""
	print("Benchmark Text Classification Pipeline Initialized")

	required_paths = [
		Path("outputs/logs"),
		Path("outputs/models"),
		Path("outputs/plots"),
		Path("results"),
	]

	for path in required_paths:
		path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
	main()

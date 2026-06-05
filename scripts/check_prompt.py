import sys
from pathlib import Path

# Ensure repository root is on PYTHONPATH for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.few_shot import FewShotClassifier


m = FewShotClassifier()
prompt = m.build_prompt('NASA launches a new satellite', m.get_examples(16))
print(len(m.tokenizer(prompt).input_ids))

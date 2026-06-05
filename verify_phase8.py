import os

checks = {
    "1. DataCollatorWithPadding import": "DataCollatorWithPadding" in open("src/models/fine_tune.py").read(),
    "2. data_collator in Trainer init": "data_collator=data_collator" in open("src/models/fine_tune.py").read(),
    "3. set_format('torch') removed": "set_format('torch')" not in open("src/models/fine_tune.py").read() and 'set_format("torch")' not in open("src/models/fine_tune.py").read(),
    "4. run_finetune_smoke_test exists": "def run_finetune_smoke_test" in open("src/experiments/run_finetune.py").read(),
    "5. numpy==1.26.4 in requirements": "numpy==1.26.4" in open("requirements.txt").read(),
    "6. transformers==4.46.3 in requirements": "transformers==4.46.3" in open("requirements.txt").read(),
    "7. No direct numpy import in fine_tune.py": "import numpy" not in open("src/models/fine_tune.py").read() and "from numpy" not in open("src/models/fine_tune.py").read(),
}

print("=" * 80)
print("PHASE 8 FINE-TUNING PIPELINE VERIFICATION")
print("=" * 80)
print()

passed = 0
for check_name, result in checks.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {check_name}")
    if result:
        passed += 1

print()
print("=" * 80)
print(f"RESULT: {passed}/{len(checks)} checks passed")
print("=" * 80)

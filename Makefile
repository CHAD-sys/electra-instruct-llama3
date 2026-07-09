# convenience targets so I stop retyping long commands
# override the interpreter with e.g. `make test PYTHON=python3` if `python`
# isn't on your PATH (systems where only python3 exists).
PYTHON ?= python

.PHONY: data train resume chat eval test clean

data:  ## build the instruction dataset from raw docs + seed pairs
	$(PYTHON) data/build_dataset.py

train: data  ## train the LoRA adapter
	$(PYTHON) -m src.train

resume:  ## resume training from the latest checkpoint after a crash
	$(PYTHON) -m src.train --resume_from_checkpoint auto

chat:  ## interactive chat with the trained adapter
	$(PYTHON) -m src.inference

eval:  ## run held-out evaluation
	$(PYTHON) eval/evaluate.py

test:  ## run the unit tests (no GPU needed)
	$(PYTHON) -m pytest -q tests/

clean:  ## remove python caches (keeps model outputs)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

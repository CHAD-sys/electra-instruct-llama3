# convenience targets so I stop retyping long commands

.PHONY: data train resume chat eval test clean

data:  ## build the instruction dataset from raw docs + seed pairs
	python data/build_dataset.py

train: data  ## train the LoRA adapter
	python -m src.train

resume:  ## resume training from the latest checkpoint after a crash
	python -m src.train --resume_from_checkpoint auto

chat:  ## interactive chat with the trained adapter
	python -m src.inference

eval:  ## run held-out evaluation
	python eval/evaluate.py

test:  ## run the unit tests (no GPU needed)
	python -m pytest -q tests/

clean:  ## remove python caches (keeps model outputs)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

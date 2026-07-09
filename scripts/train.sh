#!/usr/bin/env bash
# Kick off training. I usually run this in tmux so I can close my laptop.
set -e

# build the dataset first if it doesn't exist yet
if [ ! -f data/electrical_instructions.jsonl ]; then
  echo "dataset not found, building it..."
  python data/build_dataset.py
fi

python -m src.train "$@"

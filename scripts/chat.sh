#!/usr/bin/env bash
# Interactive chat with the trained adapter.
set -e
python -m src.inference --adapter "${1:-outputs/llama3-electrical-lora}"

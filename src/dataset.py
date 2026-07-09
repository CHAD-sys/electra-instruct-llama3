"""Dataset loading + prompt formatting.

The data is a jsonl file where each line looks like:
    {"instruction": "...", "input": "...", "output": "..."}
`input` is optional (context like a datasheet snippet). I format everything
into the Llama 3 chat template so the model sees the same structure it was
instruction-tuned on.
"""
import json
from typing import Dict, List

from datasets import Dataset


SYSTEM_PROMPT = (
    "You are an assistant specialized in electrical and electronic components. "
    "Answer precisely using correct terminology, units, and safety practices. "
    "If a value is not determinable from the given information, say so."
)


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                # I had a couple of malformed lines early on from a bad export,
                # so I skip + warn instead of crashing the whole run.
                print(f"[dataset] skipping bad line {i}: {e}")
    return rows


def build_user_turn(example: Dict) -> str:
    instr = example["instruction"].strip()
    ctx = (example.get("input") or "").strip()
    if ctx:
        return f"{instr}\n\nContext:\n{ctx}"
    return instr


def format_chat(example: Dict, tokenizer) -> Dict:
    """Render one example into a single training string via the chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_turn(example)},
        {"role": "assistant", "content": example["output"].strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def load_dataset(path: str, tokenizer) -> Dataset:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"No usable examples found in {path}")
    ds = Dataset.from_list(rows)
    ds = ds.map(lambda ex: format_chat(ex, tokenizer), remove_columns=ds.column_names)
    return ds

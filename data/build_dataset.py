"""Turn raw technical docs into instruction->response training pairs.

The idea: I collected component datasheets, application notes and a few
electronics reference pages (plain text under data/raw/). This script chunks
them and generates templated Q/A pairs so I don't have to hand-write thousands
of examples. It's deliberately simple + deterministic so the dataset is
reproducible.

    python data/build_dataset.py --raw_dir data/raw --out data/electrical_instructions.jsonl

For the bootstrap set I also mixed in some hand-written pairs
(data/seed_pairs.jsonl) which are the higher quality ones.
"""
import argparse
import glob
import json
import os
import random
import re

random.seed(7)

# Very light heuristic templates. Each raw doc is expected to start with a
# component name line, then key: value spec lines, then free-text notes.
QUESTION_TEMPLATES = [
    "What is the {field} of the {component}?",
    "Give me the {field} for a {component}.",
    "I'm reading a datasheet — what does the {field} of the {component} mean?",
]

EXPLAIN_TEMPLATES = [
    "Explain how a {component} works.",
    "In simple terms, what is a {component} used for?",
    "When would I choose a {component} in a circuit?",
]


def parse_doc(text: str):
    """Return (component_name, {field: value}, notes_paragraphs)."""
    lines = [l.rstrip() for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines:
        return None, {}, []

    component = lines[0].strip().lstrip("# ").strip()
    fields = {}
    notes = []
    for line in lines[1:]:
        m = re.match(r"^([A-Za-z][A-Za-z0-9 /()\-]+):\s*(.+)$", line)
        if m:
            fields[m.group(1).strip().lower()] = m.group(2).strip()
        else:
            notes.append(line.strip())
    return component, fields, notes


def make_pairs(component, fields, notes):
    pairs = []
    for field, value in fields.items():
        q = random.choice(QUESTION_TEMPLATES).format(field=field, component=component)
        a = f"The {field} of the {component} is {value}."
        pairs.append({"instruction": q, "input": "", "output": a})

    if notes:
        context = " ".join(notes)
        q = random.choice(EXPLAIN_TEMPLATES).format(component=component)
        # keep the answer grounded in the doc's own notes
        pairs.append({"instruction": q, "input": context[:800], "output": context})

    return pairs


def load_seed(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--seed_pairs", default="data/seed_pairs.jsonl")
    ap.add_argument("--out", default="data/electrical_instructions.jsonl")
    args = ap.parse_args()

    all_pairs = load_seed(args.seed_pairs)
    print(f"[build] loaded {len(all_pairs)} hand-written seed pairs")

    docs = sorted(glob.glob(os.path.join(args.raw_dir, "*.txt")))
    for path in docs:
        with open(path, encoding="utf-8") as f:
            component, fields, notes = parse_doc(f.read())
        if component:
            all_pairs.extend(make_pairs(component, fields, notes))

    # de-dup on (instruction, output)
    seen = set()
    deduped = []
    for p in all_pairs:
        key = (p["instruction"], p["output"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    random.shuffle(deduped)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[build] wrote {len(deduped)} pairs -> {args.out} "
          f"(from {len(docs)} raw docs)")


if __name__ == "__main__":
    main()

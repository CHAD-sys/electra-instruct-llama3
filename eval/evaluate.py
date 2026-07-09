"""Quick held-out evaluation of the fine-tuned adapter.

I'm not claiming this is a rigorous benchmark - it's a lightweight check I used
to see whether the LoRA actually helped vs the stock model. It computes:
  * token-level ROUGE-L against reference answers
  * a keyword-hit rate (does the answer mention the expected units/terms?)

    python eval/evaluate.py --adapter outputs/llama3-electrical-lora \
        --test data/eval_held_out.jsonl
"""
import argparse
import json

from src.inference import load, generate
from eval.metrics import rouge_l, keyword_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/llama3-electrical-lora")
    ap.add_argument("--test", default="data/eval_held_out.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.adapter)

    rows = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    rouge_scores, kw_scores = [], []
    for ex in rows:
        user = ex["instruction"]
        if ex.get("input"):
            user += "\n\nContext:\n" + ex["input"]
        pred = generate(model, tokenizer, user, max_new_tokens=256)
        r = rouge_l(pred, ex["output"])
        k = keyword_hit(pred, ex.get("keywords", []))
        rouge_scores.append(r)
        kw_scores.append(k)
        print(f"- ROUGE-L={r:.2f} kw={k:.2f} | {ex['instruction'][:60]}")

    n = len(rows)
    print("\n=== summary ===")
    print(f"examples     : {n}")
    print(f"mean ROUGE-L : {sum(rouge_scores)/n:.3f}")
    print(f"mean keyword : {sum(kw_scores)/n:.3f}")


if __name__ == "__main__":
    main()

"""Held-out evaluation of the fine-tuned adapter.

I'm not claiming this is a rigorous benchmark - it's a lightweight check I used
to see whether the LoRA actually helped vs the stock model. It computes:
  * token-level ROUGE-L against reference answers
  * a keyword-hit rate (does the answer mention the expected units/terms?)

    # just the fine-tune:
    python eval/evaluate.py --adapter outputs/llama3-electrical-lora

    # compare fine-tune vs the stock base model in one go:
    python eval/evaluate.py --adapter outputs/llama3-electrical-lora --compare

Results are printed and also written to eval/results.json.
"""
import argparse
import json
import os

from src.inference import load, generate
from eval.metrics import rouge_l, keyword_hit


def load_test(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def score_model(rows, adapter_dir, label, max_new_tokens=256):
    print(f"\n### evaluating: {label}")
    model, tokenizer = load(adapter_dir)
    rouge_scores, kw_scores = [], []
    for ex in rows:
        user = ex["instruction"]
        if ex.get("input"):
            user += "\n\nContext:\n" + ex["input"]
        pred = generate(model, tokenizer, user, max_new_tokens=max_new_tokens)
        r = rouge_l(pred, ex["output"])
        k = keyword_hit(pred, ex.get("keywords", []))
        rouge_scores.append(r)
        kw_scores.append(k)
        print(f"- ROUGE-L={r:.2f} kw={k:.2f} | {ex['instruction'][:60]}")

    n = len(rows)
    result = {
        "label": label,
        "examples": n,
        "mean_rouge_l": round(sum(rouge_scores) / n, 4),
        "mean_keyword_hit": round(sum(kw_scores) / n, 4),
    }
    # free the model before loading the next one (base-vs-adapter can't both fit)
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/llama3-electrical-lora")
    ap.add_argument("--test", default="data/eval_held_out.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare", action="store_true",
                    help="also evaluate the stock base model for comparison")
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    rows = load_test(args.test, args.limit)

    results = [score_model(rows, args.adapter, "fine-tuned (LoRA)")]
    if args.compare:
        results.append(score_model(rows, None, "base (Llama-3-8B-Instruct)"))

    print("\n=== summary ===")
    for r in results:
        print(f"{r['label']:<32} ROUGE-L={r['mean_rouge_l']:.3f} "
              f"keyword={r['mean_keyword_hit']:.3f}  (n={r['examples']})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

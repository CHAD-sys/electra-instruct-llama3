"""Tiny Gradio chat UI for the fine-tuned model.

    pip install gradio
    python app.py --adapter outputs/llama3-electrical-lora

Then open the local URL it prints. Handy for showing the thing off without
making people use the terminal.
"""
import argparse

import gradio as gr

from src.inference import load, generate

EXAMPLES = [
    "How do I size the series resistor for a red LED on 5V at 10 mA?",
    "What do the color bands on a 4-band resistor mean?",
    "Can I drive an IRF540N directly from a 3.3V microcontroller pin?",
    "Why put a decoupling capacitor next to an IC power pin?",
]


def build_ui(model, tokenizer, max_new_tokens):
    def respond(message, history):
        return generate(model, tokenizer, message, max_new_tokens=max_new_tokens)

    return gr.ChatInterface(
        fn=respond,
        title="⚡ Electrical Components Assistant (Llama 3 8B + LoRA)",
        description=(
            "A Llama 3 8B model fine-tuned with QLoRA on electrical/electronic "
            "component Q&A. Answers are best-effort — verify against a real "
            "datasheet before you solder anything."
        ),
        examples=EXAMPLES,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/llama3-electrical-lora")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--share", action="store_true", help="create a public link")
    args = ap.parse_args()

    print("[app] loading model, this takes a minute...")
    model, tokenizer = load(args.adapter)
    ui = build_ui(model, tokenizer, args.max_new_tokens)
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()

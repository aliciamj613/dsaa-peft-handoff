"""Evaluate a (base model + optional PEFT adapter) on DialogSum.

Loads the held-out test JSONL written by `scripts/preprocess/build_dialogsum_budget.py`,
generates a summary per dialogue with greedy decoding, scores it against the
reference with ROUGE-L, and writes both the averaged metric and per-example
predictions to a JSON file. `--adapter_path none` (or omitting it) evaluates
the raw base model so we have a zero-shot baseline.
"""

import os
import json
import math
import argparse
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_examples", type=int, default=None)
    return p.parse_args()


def build_prompt(dialogue: str) -> str:
    # Must match the instruction/response template used during training so the
    # adapter sees the same prefix distribution it was fine-tuned on.
    return (
        "### Instruction:\n"
        "Summarize the following dialogue concisely.\n\n"
        f"### Input:\n{dialogue}\n\n"
        "### Response:\n"
    )


def chunk_list(lst: List, batch_size: int):
    """Yield successive `batch_size`-sized slices from `lst`."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        # Many decoder-only LMs (Qwen, Gemma, LLaMA) ship without a pad token;
        # reuse EOS so right/left padding works during batched generation.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )
    # "none" lets the launcher template pass a single value for both
    # baseline (no adapter) and adapter runs.
    if args.adapter_path and str(args.adapter_path).lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    ds = load_dataset("json", data_files=args.test_file)["train"]
    if args.max_examples is not None:
        n = min(args.max_examples, len(ds))
        ds = ds.select(range(n))

    records = list(ds)
    total = len(records)
    print(f"Loaded {total} evaluation examples")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    preds = []
    rouge_scores = []

    num_batches = math.ceil(total / args.batch_size)

    for batch in tqdm(chunk_list(records, args.batch_size), total=num_batches, desc="Evaluating DialogSum"):
        prompts = [build_prompt(ex["dialogue"]) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for ex, decoded, prompt in zip(batch, decoded_batch, prompts):
            # `generate` echoes the prompt; strip it so the metric only scores
            # the newly generated continuation. If decoding lost an exact prefix
            # match (e.g. due to tokenizer round-trip), fall back to using the
            # full decoded string.
            if decoded.startswith(prompt):
                pred_text = decoded[len(prompt):].strip()
            else:
                pred_text = decoded.strip()

            gold = ex["summary"].strip()
            rougeL = scorer.score(gold, pred_text)["rougeL"].fmeasure
            rouge_scores.append(rougeL)

            preds.append({
                "dialogue": ex["dialogue"],
                "gold_summary": gold,
                "prediction_text": pred_text,
                "rougeL": rougeL,
            })

    avg_rougeL = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rougeL": avg_rougeL,
                "n": total,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "predictions": preds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"ROUGE-L = {avg_rougeL:.4f} over {total} examples")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()

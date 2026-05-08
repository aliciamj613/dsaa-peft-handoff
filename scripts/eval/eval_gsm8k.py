import os
import re
import json
import math
import argparse
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_examples", type=int, default=None,
                   help="Only evaluate the first N examples for quick debugging")
    return p.parse_args()


def extract_final_number(text: str):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def build_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        "Solve the following math word problem. Show the reasoning briefly and end with the final answer.\n\n"
        f"### Input:\n{question}\n\n"
        "### Response:\n"
    )


def chunk_list(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )
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

    preds = []
    correct = 0

    num_batches = math.ceil(total / args.batch_size)

    for batch in tqdm(chunk_list(records, args.batch_size), total=num_batches, desc="Evaluating GSM8K"):
        prompts = [build_prompt(ex["question"]) for ex in batch]

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
            # 只保留 prompt 之后的生成部分
            if decoded.startswith(prompt):
                pred_text = decoded[len(prompt):].strip()
            else:
                pred_text = decoded.strip()

            pred_num = extract_final_number(pred_text)
            gold_num = extract_final_number(ex["answer"])

            is_correct = int(pred_num == gold_num)
            correct += is_correct

            preds.append({
                "question": ex["question"],
                "gold": ex["answer"],
                "prediction_text": pred_text,
                "pred_num": pred_num,
                "gold_num": gold_num,
                "correct": is_correct,
            })

    em = correct / total if total > 0 else 0.0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "em": em,
                "n": total,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "predictions": preds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"EM = {em:.4f} over {total} examples")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()

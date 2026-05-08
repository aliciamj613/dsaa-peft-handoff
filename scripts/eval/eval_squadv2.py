import os
import re
import json
import math
import string
import argparse
from collections import Counter
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
    p.add_argument("--max_examples", type=int, default=None)
    return p.parse_args()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()

    if len(gold_toks) == 0 and len(pred_toks) == 0:
        return 1.0
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0.0

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def build_prompt(context: str, question: str) -> str:
    return (
        "### Instruction:\n"
        "Read the context and answer the question. "
        "If the question is unanswerable from the context, output exactly: unanswerable. "
        "Otherwise, output only the shortest answer span. Do not explain.\n\n"
        f"### Input:\nContext: {context}\n\nQuestion: {question}\n\n"
        "### Response:\n"
    )


def chunk_list(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )
    if args.adapter_path and str(args.adapter_path).lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    ds = load_dataset("json", data_files=args.test_file)["train"]
    if args.max_examples is not None:
        n = min(args.max_examples, len(ds))
        ds = ds.select(range(n))

    records = list(ds)
    total = len(records)
    print(f"Loaded {total} evaluation examples")

    preds = []
    em_scores = []
    f1_scores = []

    num_batches = math.ceil(total / args.batch_size)

    for batch in tqdm(chunk_list(records, args.batch_size), total=num_batches, desc="Evaluating SQuAD v2"):
        prompts = [build_prompt(ex["context"], ex["question"]) for ex in batch]

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
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for ex, decoded, prompt in zip(batch, decoded_batch, prompts):
            if decoded.startswith(prompt):
                pred_text = decoded[len(prompt):].strip()
            else:
                pred_text = decoded.strip()

            gold = ex["answer"].strip()

            em = compute_exact(gold, pred_text)
            f1 = compute_f1(gold, pred_text)
            em_scores.append(em)
            f1_scores.append(f1)

            preds.append({
                "id": ex["id"],
                "question": ex["question"],
                "gold_answer": gold,
                "prediction_text": pred_text,
                "em": em,
                "f1": f1,
            })

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "em": avg_em,
                "f1": avg_f1,
                "n": total,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "predictions": preds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"EM = {avg_em:.4f}, F1 = {avg_f1:.4f} over {total} examples")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()

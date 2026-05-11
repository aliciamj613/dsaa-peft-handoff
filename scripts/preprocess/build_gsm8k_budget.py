"""Build budgeted GSM8K training splits in instruction-response format.

Shuffles the full train split once (fixed seed) and writes the first N rows
to `train_<N>.jsonl` for each budget N. The test split is dumped verbatim so
every fine-tuned checkpoint is evaluated on the same questions.
"""

import os
import json
import random
from datasets import load_dataset

OUT_DIR = os.path.expanduser("~/projects/dsaa_peft/data/gsm8k")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BUDGETS = [128, 512, 2048]

def format_example(ex):
    """Wrap a raw GSM8K row in the shared instruction-response template.

    `text` is the full training string; `question` / `answer` are kept so
    that the evaluator can recover the gold answer for metric computation.
    """
    question = ex["question"].strip()
    answer = ex["answer"].strip()
    text = (
        "### Instruction:\n"
        "Solve the following math word problem. Show the reasoning briefly and end with the final answer.\n\n"
        f"### Input:\n{question}\n\n"
        "### Response:\n"
        f"{answer}"
    )
    return {"text": text, "question": question, "answer": answer}

def main():
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = [format_example(x) for x in ds["train"]]
    test_ds = [format_example(x) for x in ds["test"]]

    # Shuffle once so smaller budgets are prefixes of larger ones; this lets
    # us attribute performance differences to scale rather than to which
    # examples happened to be drawn.
    random.seed(SEED)
    random.shuffle(train_ds)

    for b in BUDGETS:
        subset = train_ds[:b]
        out_path = os.path.join(OUT_DIR, f"train_{b}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {out_path}: {len(subset)} examples")

    test_path = os.path.join(OUT_DIR, "test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for row in test_ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {test_path}: {len(test_ds)} examples")

if __name__ == "__main__":
    main()
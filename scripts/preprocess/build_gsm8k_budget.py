import os
import json
import random
from datasets import load_dataset

OUT_DIR = os.path.expanduser("~/projects/dsaa_peft/data/gsm8k")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BUDGETS = [128, 512, 2048]

def format_example(ex):
    question = ex["question"].strip()
    answer = ex["answer"].strip()
    # instruction-response format
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
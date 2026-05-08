import os
import json
import random
from datasets import load_dataset

OUT_DIR = os.path.expanduser("~/projects/dsaa_peft/data/squad_v2")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BUDGETS = [128, 512, 2048]

def get_answer(ex):
    answers = ex["answers"]["text"]
    if len(answers) == 0:
        return "unanswerable"
    return answers[0].strip()

def format_example(ex):
    context = ex["context"].strip()
    question = ex["question"].strip()
    answer = get_answer(ex)
    text = (
        "### Instruction:\n"
        "Answer the question based on the given context. If the answer is not in the context, output 'unanswerable'.\n\n"
        f"### Input:\nContext: {context}\n\nQuestion: {question}\n\n"
        "### Response:\n"
        f"{answer}"
    )
    return {
        "text": text,
        "context": context,
        "question": question,
        "answer": answer,
        "id": ex["id"],
    }

def main():
    ds = load_dataset("rajpurkar/squad_v2")
    train_ds = [format_example(x) for x in ds["train"]]
    val_ds = [format_example(x) for x in ds["validation"]]

    random.seed(SEED)
    random.shuffle(train_ds)

    for b in BUDGETS:
        subset = train_ds[:b]
        out_path = os.path.join(OUT_DIR, f"train_{b}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {out_path}: {len(subset)} examples")

    val_path = os.path.join(OUT_DIR, "validation.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for row in val_ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {val_path}: {len(val_ds)} examples")

if __name__ == "__main__":
    main()

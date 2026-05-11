"""Build budgeted DialogSum training splits in instruction-response format.

The full DialogSum train split is shuffled once (with a fixed seed for
reproducibility) and then the first N examples are written to
`train_<N>.jsonl` for each budget N. The validation and test splits are
written out verbatim so all downstream evaluations score against the same
references regardless of budget.
"""

import os
import json
import random
from datasets import load_dataset

OUT_DIR = os.path.expanduser("~/projects/dsaa_peft/data/dialogsum")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BUDGETS = [128, 512, 2048]

def format_example(ex):
    """Wrap a raw DialogSum row in the standard instruction template.

    The `text` field is the full prompt+response string that the language
    model is trained on. The original dialogue/summary are kept alongside so
    that evaluation scripts can recover them without re-parsing `text`.
    """
    dialogue = ex["dialogue"].strip()
    summary = ex["summary"].strip()
    text = (
        "### Instruction:\n"
        "Summarize the following dialogue concisely.\n\n"
        f"### Input:\n{dialogue}\n\n"
        "### Response:\n"
        f"{summary}"
    )
    return {
        "text": text,
        "dialogue": dialogue,
        "summary": summary,
    }

def main():
    ds = load_dataset("knkarthick/dialogsum")
    train_ds = [format_example(x) for x in ds["train"]]
    val_ds = [format_example(x) for x in ds["validation"]]
    test_ds = [format_example(x) for x in ds["test"]]

    # Shuffle once so the same prefix is reused across budgets: the
    # 128-example subset is contained in the 512-example subset, etc.
    random.seed(SEED)
    random.shuffle(train_ds)

    for b in BUDGETS:
        subset = train_ds[:b]
        out_path = os.path.join(OUT_DIR, f"train_{b}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {out_path}: {len(subset)} examples")

    for split_name, split_data in [("validation", val_ds), ("test", test_ds)]:
        out_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {out_path}: {len(split_data)} examples")

if __name__ == "__main__":
    main()

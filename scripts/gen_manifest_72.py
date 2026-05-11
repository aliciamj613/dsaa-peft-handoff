"""Generate the run manifest CSV for the 72-run debug sweep.

Each row in the manifest represents one (model x method x task x budget x seed)
configuration. The companion `submit_train_batch_72.sh` / `submit_eval_batch_72.sh`
scripts iterate over rows in this CSV and submit a SLURM job per row.
"""

import csv
from pathlib import Path

out = Path("/hpc2hdd/home/mliu954/projects/dsaa_peft/manifests/main_debug_72.csv")
out.parent.mkdir(parents=True, exist_ok=True)

models = [
    ("qwen", "/hpc2hdd/home/mliu954/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    ("gemma", "/hpc2hdd/home/mliu954/hf_models/gemma-2-2b-it"),
]
methods = ["lora", "qlora", "dora", "ia3"]
tasks = ["gsm8k", "dialogsum", "squad_v2"]
budgets = [128, 512, 1024]
seed = 42

# Per-method learning rates chosen from the pilot sweep. IA3 needs a larger LR
# because it has far fewer trainable parameters than LoRA-family methods.
lr_map = {
    "lora": "2e-4",
    "qlora": "1e-4",
    "dora": "1e-4",
    "ia3": "1e-3",
}

rows = []
for model_tag, model_name in models:
    for method in methods:
        for task in tasks:
            for budget in budgets:
                run_id = f"{model_tag}_{task}_{method}_b{budget}_s{seed}_dbg72"
                epochs = 2
                max_length = 768

                # DoRA on the largest SQuAD v2 budget pushes activation memory
                # past the single-GPU debug-partition limit at 768 tokens, so
                # shrink the sequence length for just that cell.
                if method == "dora" and task == "squad_v2" and budget == 1024:
                    max_length = 640

                rows.append({
                    "run_id": run_id,
                    "model_tag": model_tag,
                    "model_name": model_name,
                    "method": method,
                    "task": task,
                    "budget": budget,
                    "seed": seed,
                    "lr": lr_map[method],
                    "epochs": epochs,
                    "max_length": max_length,
                })

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_id", "model_tag", "model_name", "method", "task",
            "budget", "seed", "lr", "epochs", "max_length"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("saved:", out)
print("rows:", len(rows))

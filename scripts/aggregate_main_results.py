import os
import json
import glob
import csv

BASE = "/hpc2hdd/home/mliu954/projects/dsaa_peft/results/main"
OUT_CSV = "/hpc2hdd/home/mliu954/projects/dsaa_peft/results/main_summary.csv"

files = sorted(glob.glob(os.path.join(BASE, "*_meta.json")))

rows = []
for fp in files:
    try:
        x = json.load(open(fp))
    except Exception as e:
        print(f"skip broken file: {fp} ({e})")
        continue

    run_id = x.get("run_id")
    model_tag = x.get("model_tag")
    model_name = x.get("model_name")
    task = x.get("task")
    method = x.get("method")
    budget = x.get("budget")
    seed = x.get("seed")
    lr = x.get("lr")

    train = x.get("train_summary", {})
    evals = x.get("eval_summary", {})

    peak_vram_gb = train.get("peak_vram_gb")
    total_train_time_min = train.get("total_train_time_min")
    trainable_ratio = train.get("trainable_ratio")

    em = evals.get("em")
    f1 = evals.get("f1")
    rougeL = evals.get("rougeL")
    n = evals.get("n")

    # 统一一个主指标列，方便后续画图/排序
    if task == "gsm8k":
        main_metric = em
        main_metric_name = "em"
    elif task == "dialogsum":
        main_metric = rougeL
        main_metric_name = "rougeL"
    elif task == "squad_v2":
        main_metric = f1 if f1 is not None else em
        main_metric_name = "f1"
    else:
        main_metric = None
        main_metric_name = ""

    rows.append([
        run_id,
        model_tag,
        model_name,
        task,
        method,
        budget,
        seed,
        lr,
        main_metric_name,
        main_metric,
        em,
        f1,
        rougeL,
        n,
        peak_vram_gb,
        total_train_time_min,
        trainable_ratio,
        fp,
    ])

rows.sort(key=lambda r: (r[1], r[3], r[4], int(r[5]), int(r[6])))

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_id",
        "model_tag",
        "model_name",
        "task",
        "method",
        "budget",
        "seed",
        "lr",
        "main_metric_name",
        "main_metric",
        "em",
        "f1",
        "rougeL",
        "n",
        "peak_vram_gb",
        "total_train_time_min",
        "trainable_ratio",
        "meta_file",
    ])
    writer.writerows(rows)

print(f"saved: {OUT_CSV}")
print(f"total rows: {len(rows)}")

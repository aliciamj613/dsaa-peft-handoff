#!/bin/bash
# Pilot LR sweep for Qwen on GSM8K at budget=512. Uses a SLURM array job
# (8 tasks) where each array index picks one (METHOD, LR) configuration from
# CONFIGS. Result: 8 fine-tuned adapters + per-run eval/meta JSON files so
# we can pick the best LR per PEFT method for the main sweep.
#SBATCH -J peft_pilot_qwen_gsm8k
#SBATCH -p long_gpu 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o ~/projects/dsaa_peft/logs/%x_%A_%a.out
#SBATCH -e ~/projects/dsaa_peft/logs/%x_%A_%a.err
#SBATCH --array=0-7

set -euo pipefail

module load anaconda3
module load cuda/12.8
conda activate peft_llm

cd ~/projects/dsaa_peft

mkdir -p outputs
mkdir -p results/pilot
mkdir -p logs

CONFIGS=(
"lora  1e-4 qwen_gsm8k_lora_b512_s42_lr1e4"
"lora  2e-4 qwen_gsm8k_lora_b512_s42_lr2e4"
"qlora 1e-4 qwen_gsm8k_qlora_b512_s42_lr1e4"
"qlora 2e-4 qwen_gsm8k_qlora_b512_s42_lr2e4"
"dora  1e-4 qwen_gsm8k_dora_b512_s42_lr1e4"
"dora  2e-4 qwen_gsm8k_dora_b512_s42_lr2e4"
"ia3   5e-4 qwen_gsm8k_ia3_b512_s42_lr5e4"
"ia3   1e-3 qwen_gsm8k_ia3_b512_s42_lr1e3"
)

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
METHOD=$(echo $CFG | awk '{print $1}')
LR=$(echo $CFG | awk '{print $2}')
RUNID=$(echo $CFG | awk '{print $3}')

OUTDIR=~/projects/dsaa_peft/outputs/${RUNID}
EVALJSON=~/projects/dsaa_peft/results/pilot/${RUNID}_eval200.json
METAJSON=~/projects/dsaa_peft/results/pilot/${RUNID}_meta.json

echo "===== JOB INFO ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "RUNID=$RUNID"
echo "METHOD=$METHOD"
echo "LR=$LR"
echo "START_TIME=$(date)"
nvidia-smi

python scripts/train/train_peft.py \
  --method ${METHOD} \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --train_file ~/projects/dsaa_peft/data/gsm8k/train_512.jsonl \
  --output_dir ${OUTDIR} \
  --learning_rate ${LR} \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --seed 42

python scripts/eval/eval_gsm8k.py \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --adapter_path ${OUTDIR} \
  --test_file ~/projects/dsaa_peft/data/gsm8k/test.jsonl \
  --output_file ${EVALJSON} \
  --max_examples 200 \
  --batch_size 4 \
  --max_new_tokens 32

python - <<PY
import os, json, glob

runid = "${RUNID}"
outdir = os.path.expanduser("${OUTDIR}")
evaljson = os.path.expanduser("${EVALJSON}")
metajson = os.path.expanduser("${METAJSON}")

train_summary_path = os.path.join(outdir, "train_summary.json")

data = {
    "run_id": runid,
    "method": "${METHOD}",
    "lr": "${LR}",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "task": "gsm8k",
    "budget": 512,
    "seed": 42,
}

if os.path.exists(train_summary_path):
    train_summary = json.load(open(train_summary_path))
    data["train_summary"] = train_summary

if os.path.exists(evaljson):
    eval_summary = json.load(open(evaljson))
    data["eval_summary"] = {
        "em": eval_summary.get("em"),
        "n": eval_summary.get("n"),
        "batch_size": eval_summary.get("batch_size"),
        "max_new_tokens": eval_summary.get("max_new_tokens"),
    }

with open(metajson, "w") as f:
    json.dump(data, f, indent=2)

print("Saved meta summary to", metajson)
PY

echo "END_TIME=$(date)"
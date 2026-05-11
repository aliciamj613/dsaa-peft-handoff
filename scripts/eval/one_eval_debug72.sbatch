#!/bin/bash
# One-row evaluation step of the 72-run debug sweep. Dispatched per manifest
# row by submit_eval_batch_72.sh; picks the task-specific eval script based
# on TASK, then writes a *_meta.json that merges the train and eval summaries
# for that run_id. Skips work if the eval result already exists.
#SBATCH -J one_eval_dbg72
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /hpc2hdd/home/mliu954/projects/dsaa_peft/logs/72/%x_%j.out
#SBATCH -e /hpc2hdd/home/mliu954/projects/dsaa_peft/logs/72/%x_%j.err

set -eo pipefail

module load anaconda3
module load cuda/12.8

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate peft_llm
set -u

# Force fully-offline mode: HF cache must already contain the base model and
# tokenizer; otherwise the compute node will hang trying to reach the hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ROOT=/hpc2hdd/home/mliu954/projects/dsaa_peft
cd ${ROOT}

: "${RUN_ID:?Need RUN_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${TASK:?Need TASK}"

ADAPTER_PATH="${ROOT}/outputs_72/${RUN_ID}"
OUTFILE="${ROOT}/results/main_debug_72/${RUN_ID}_eval50.json"
METAFILE="${ROOT}/results/main_debug_72/${RUN_ID}_meta.json"

if [ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]; then
  echo "Missing adapter: ${ADAPTER_PATH}/adapter_model.safetensors"
  exit 1
fi

if [ -f "${OUTFILE}" ]; then
  echo "Eval result already exists, skip: ${RUN_ID}"
  exit 0
fi

if [ "${TASK}" = "gsm8k" ]; then
  python ${ROOT}/scripts/eval/eval_gsm8k.py \
    --base_model "${MODEL_NAME}" \
    --adapter_path "${ADAPTER_PATH}" \
    --test_file ${ROOT}/data/gsm8k/test.jsonl \
    --output_file "${OUTFILE}" \
    --max_examples 50 \
    --batch_size 4 \
    --max_new_tokens 24

elif [ "${TASK}" = "dialogsum" ]; then
  python ${ROOT}/scripts/eval/eval_dialogsum.py \
    --base_model "${MODEL_NAME}" \
    --adapter_path "${ADAPTER_PATH}" \
    --test_file ${ROOT}/data/dialogsum/test.jsonl \
    --output_file "${OUTFILE}" \
    --max_examples 50 \
    --batch_size 4 \
    --max_new_tokens 48

elif [ "${TASK}" = "squad_v2" ]; then
  python ${ROOT}/scripts/eval/eval_squadv2.py \
    --base_model "${MODEL_NAME}" \
    --adapter_path "${ADAPTER_PATH}" \
    --test_file ${ROOT}/data/squad_v2/validation.jsonl \
    --output_file "${OUTFILE}" \
    --max_examples 50 \
    --batch_size 4 \
    --max_new_tokens 24
else
  echo "Unknown TASK=${TASK}"
  exit 1
fi

python - <<PY
import os, json

root = "/hpc2hdd/home/mliu954/projects/dsaa_peft"
run_id = "${RUN_ID}"
task = "${TASK}"
model_name = "${MODEL_NAME}"

adapter_path = os.path.join(root, "outputs_72", run_id)
outfile = os.path.join(root, "results", "main_debug_72", f"{run_id}_eval50.json")
metafile = os.path.join(root, "results", "main_debug_72", f"{run_id}_meta.json")

meta = {
    "run_id": run_id,
    "task": task,
    "model_name": model_name,
}

train_summary = os.path.join(adapter_path, "train_summary.json")
if os.path.exists(train_summary):
    meta["train_summary"] = json.load(open(train_summary))

if os.path.exists(outfile):
    meta["eval_summary"] = json.load(open(outfile))

with open(metafile, "w") as f:
    json.dump(meta, f, indent=2)

print("Saved meta:", metafile)
PY

echo "EVAL DONE $(date)"

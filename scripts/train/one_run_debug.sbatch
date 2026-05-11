#!/bin/bash
# Train + evaluate a single (model, method, task, budget, seed) configuration
# end-to-end on the debug partition. All run parameters are read from the
# environment so the same script can be reused across quick smoke tests.
#SBATCH -J one_run_debug
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /hpc2hdd/home/mliu954/projects/dsaa_peft/logs/%x_%j.out
#SBATCH -e /hpc2hdd/home/mliu954/projects/dsaa_peft/logs/%x_%j.err

set -eo pipefail

module load anaconda3
module load cuda/12.8

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate peft_llm
set -u

ROOT=/hpc2hdd/home/mliu954/projects/dsaa_peft
cd ${ROOT}

mkdir -p ${ROOT}/logs
mkdir -p ${ROOT}/outputs
mkdir -p ${ROOT}/results/main_debug

: "${MODEL_TAG:?Need MODEL_TAG}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${METHOD:?Need METHOD}"
: "${TASK:?Need TASK}"
: "${BUDGET:?Need BUDGET}"
: "${SEED:?Need SEED}"
: "${LR:?Need LR}"

RUNID="${MODEL_TAG}_${TASK}_${METHOD}_b${BUDGET}_s${SEED}_dbg"
OUTDIR="${ROOT}/outputs/${RUNID}"
EVALJSON="${ROOT}/results/main_debug/${RUNID}_eval50.json"
METAJSON="${ROOT}/results/main_debug/${RUNID}_meta.json"

rm -rf "${OUTDIR}"
rm -f "${EVALJSON}" "${METAJSON}"

if [ "${TASK}" = "gsm8k" ]; then
  TRAIN_FILE="${ROOT}/data/gsm8k/train_${BUDGET}.jsonl"
  TEST_FILE="${ROOT}/data/gsm8k/test.jsonl"
  MAX_NEW_TOKENS=24
elif [ "${TASK}" = "dialogsum" ]; then
  TRAIN_FILE="${ROOT}/data/dialogsum/train_${BUDGET}.jsonl"
  TEST_FILE="${ROOT}/data/dialogsum/test.jsonl"
  MAX_NEW_TOKENS=48
elif [ "${TASK}" = "squad_v2" ]; then
  TRAIN_FILE="${ROOT}/data/squad_v2/train_${BUDGET}.jsonl"
  TEST_FILE="${ROOT}/data/squad_v2/validation.jsonl"
  MAX_NEW_TOKENS=24
else
  echo "Unknown TASK=${TASK}"
  exit 1
fi

echo "===== JOB INFO ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "METHOD=${METHOD}"
echo "TASK=${TASK}"
echo "BUDGET=${BUDGET}"
echo "SEED=${SEED}"
echo "LR=${LR}"
echo "RUNID=${RUNID}"
echo "START_TIME=$(date)"
which python
python -V
nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available in this job environment.")
PY

python ${ROOT}/scripts/train/train_peft.py \
  --method "${METHOD}" \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTDIR}" \
  --learning_rate "${LR}" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 768 \
  --seed "${SEED}"

if [ "${TASK}" = "gsm8k" ]; then
  python ${ROOT}/scripts/eval/eval_gsm8k.py \
    --base_model "${MODEL_NAME}" \
    --adapter_path "${OUTDIR}" \
    --test_file "${TEST_FILE}" \
    --output_file "${EVALJSON}" \
    --max_examples 50 \
    --batch_size 4 \
    --max_new_tokens ${MAX_NEW_TOKENS}
elif [ "${TASK}" = "dialogsum" ]; then
  python ${ROOT}/scripts/eval/eval_dialogsum.py \
    --base_model "${MODEL_NAME}" \
    --adapter_path "${OUTDIR}" \
    --test_file "${TEST_FILE}" \
    --output_file "${EVALJSON}" \
    --max_examples 50 \
    --batch_size 4 \
    --max_new_tokens ${MAX_NEW_TOKENS}
elif [ "${TASK}" = "squad_v2" ]; then
  python ${ROOT}/scripts/eval/eval_squadv2.py \
    --base_model "${MODEL_NAME}" \
    --adapter_path "${OUTDIR}" \
    --test_file "${TEST_FILE}" \
    --output_file "${EVALJSON}" \
    --max_examples 50 \
    --batch_size 4 \
    --max_new_tokens ${MAX_NEW_TOKENS}
fi

export RUNID OUTDIR EVALJSON METAJSON MODEL_TAG MODEL_NAME TASK METHOD BUDGET SEED LR

python - <<'PY'
import os, json

runid = os.environ["RUNID"]
outdir = os.environ["OUTDIR"]
evaljson = os.environ["EVALJSON"]
metajson = os.environ["METAJSON"]

data = {
    "run_id": runid,
    "model_tag": os.environ["MODEL_TAG"],
    "model_name": os.environ["MODEL_NAME"],
    "task": os.environ["TASK"],
    "method": os.environ["METHOD"],
    "budget": int(os.environ["BUDGET"]),
    "seed": int(os.environ["SEED"]),
    "lr": os.environ["LR"],
}

train_summary_path = os.path.join(outdir, "train_summary.json")
if os.path.exists(train_summary_path):
    data["train_summary"] = json.load(open(train_summary_path))

if os.path.exists(evaljson):
    data["eval_summary"] = json.load(open(evaljson))

with open(metajson, "w") as f:
    json.dump(data, f, indent=2)

print("Saved meta summary to", metajson)
PY

echo "DONE $(date)"

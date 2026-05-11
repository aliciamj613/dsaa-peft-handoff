#!/bin/bash
# Train one row from the 72-run debug manifest. Dispatched per manifest row
# by submit_train_batch_72.sh; reads RUN_ID / MODEL_NAME / METHOD / TASK /
# BUDGET / SEED / LR / EPOCHS / MAX_LENGTH from the environment, picks the
# matching train_<BUDGET>.jsonl file, and writes the adapter under
# outputs_72/${RUN_ID}/. Skips work if the adapter already exists.
#SBATCH -J one_train_dbg72
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

# Force fully-offline mode; the HF cache must already contain the base model.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ROOT=/hpc2hdd/home/mliu954/projects/dsaa_peft
cd ${ROOT}

: "${RUN_ID:?Need RUN_ID}"
: "${MODEL_TAG:?Need MODEL_TAG}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${METHOD:?Need METHOD}"
: "${TASK:?Need TASK}"
: "${BUDGET:?Need BUDGET}"
: "${SEED:?Need SEED}"
: "${LR:?Need LR}"
: "${EPOCHS:?Need EPOCHS}"
: "${MAX_LENGTH:?Need MAX_LENGTH}"

OUTDIR="${ROOT}/outputs_72/${RUN_ID}"

if [ -f "${OUTDIR}/adapter_model.safetensors" ]; then
  echo "Adapter already exists, skip: ${RUN_ID}"
  exit 0
fi

mkdir -p "${OUTDIR}"

if [ "${TASK}" = "gsm8k" ]; then
  TRAIN_FILE="${ROOT}/data/gsm8k/train_${BUDGET}.jsonl"
elif [ "${TASK}" = "dialogsum" ]; then
  TRAIN_FILE="${ROOT}/data/dialogsum/train_${BUDGET}.jsonl"
elif [ "${TASK}" = "squad_v2" ]; then
  TRAIN_FILE="${ROOT}/data/squad_v2/train_${BUDGET}.jsonl"
else
  echo "Unknown TASK=${TASK}"
  exit 1
fi

echo "===== TRAIN JOB INFO ====="
echo "RUN_ID=${RUN_ID}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "METHOD=${METHOD}"
echo "TASK=${TASK}"
echo "BUDGET=${BUDGET}"
echo "SEED=${SEED}"
echo "LR=${LR}"
echo "EPOCHS=${EPOCHS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "START_TIME=$(date)"
which python
python -V
nvidia-smi

python ${ROOT}/scripts/train/train_peft.py \
  --method "${METHOD}" \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTDIR}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length "${MAX_LENGTH}" \
  --seed "${SEED}"

echo "TRAIN DONE $(date)"

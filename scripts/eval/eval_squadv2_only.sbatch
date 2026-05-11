#!/bin/bash
# Single-run SQuAD v2 evaluation on the debug partition. Caller exports
# BASE_MODEL, ADAPTER_PATH, OUTFILE; this job loads that adapter on top of
# the base model and writes EM/F1 plus per-example predictions to OUTFILE.
#SBATCH -J eval_squadv2_only
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
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

: "${BASE_MODEL:?Need BASE_MODEL}"
: "${ADAPTER_PATH:?Need ADAPTER_PATH}"
: "${OUTFILE:?Need OUTFILE}"

python ${ROOT}/scripts/eval/eval_squadv2.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --test_file ${ROOT}/data/squad_v2/validation.jsonl \
  --output_file "${OUTFILE}" \
  --max_examples 50 \
  --batch_size 4 \
  --max_new_tokens 24

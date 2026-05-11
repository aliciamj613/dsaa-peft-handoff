#!/bin/bash
# Submit a contiguous slice [START..END] (0-indexed, inclusive) of the
# 72-run debug manifest to SLURM as training jobs. Each row becomes one
# `one_train_debug72.sbatch` job. Used together with submit_eval_batch_72.sh
# to drive the full sweep in two phases (train -> eval).
#
# Usage: ./submit_train_batch_72.sh <START> <END>
set -euo pipefail

START=${1:?Need START}
END=${2:?Need END}

MANIFEST=/hpc2hdd/home/mliu954/projects/dsaa_peft/manifests/main_debug_72.csv
SBATCH_FILE=/hpc2hdd/home/mliu954/projects/dsaa_peft/scripts/train/one_train_debug72.sbatch

awk -F, -v s="$START" -v e="$END" 'NR>1 && NR-1>=s && NR-1<=e {print}' "${MANIFEST}" | \
while IFS=, read -r run_id model_tag model_name method task budget seed lr epochs max_length
do
  echo "Submitting TRAIN: ${run_id}"
  sbatch --export=ALL,RUN_ID="${run_id}",MODEL_TAG="${model_tag}",MODEL_NAME="${model_name}",METHOD="${method}",TASK="${task}",BUDGET="${budget}",SEED="${seed}",LR="${lr}",EPOCHS="${epochs}",MAX_LENGTH="${max_length}" "${SBATCH_FILE}"
  sleep 1
done

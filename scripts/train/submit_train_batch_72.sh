#!/bin/bash
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

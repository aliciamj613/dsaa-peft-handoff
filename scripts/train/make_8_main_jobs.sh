#!/bin/bash
set -euo pipefail

ROOT=/hpc2hdd/home/mliu954/projects/dsaa_peft
OUTDIR=${ROOT}/scripts/train/main_jobs

mkdir -p "${OUTDIR}"
mkdir -p "${ROOT}/logs"
mkdir -p "${ROOT}/outputs"
mkdir -p "${ROOT}/results/main"

make_job () {
  local MODEL_TAG="$1"
  local MODEL_NAME="$2"
  local METHOD="$3"
  local LR="$4"
  local JOB_NAME="main_${MODEL_TAG}_${METHOD}"
  local FILE="${OUTDIR}/${JOB_NAME}.sbatch"

  cat > "${FILE}" <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p long_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH -o ${ROOT}/logs/%x_%j.out
#SBATCH -e ${ROOT}/logs/%x_%j.err

set -eo pipefail

module load anaconda3
module load cuda/12.8

set +u
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate peft_llm
set -u

cd ${ROOT}

mkdir -p ${ROOT}/logs
mkdir -p ${ROOT}/outputs
mkdir -p ${ROOT}/results/main

MODEL_TAG="${MODEL_TAG}"
MODEL_NAME="${MODEL_NAME}"
METHOD="${METHOD}"
LR="${LR}"

TASKS=("gsm8k" "dialogsum" "squad_v2")
BUDGETS=("128" "512" "2048")
SEEDS=("42" "43")

echo "===== JOB INFO ====="
echo "JOB_ID=\$SLURM_JOB_ID"
echo "PARTITION=\$SLURM_JOB_PARTITION"
echo "NODELIST=\$SLURM_JOB_NODELIST"
echo "MODEL_TAG=\$MODEL_TAG"
echo "MODEL_NAME=\$MODEL_NAME"
echo "METHOD=\$METHOD"
echo "LR=\$LR"
echo "START_TIME=\$(date)"
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

for TASK in "\${TASKS[@]}"; do
  for BUDGET in "\${BUDGETS[@]}"; do
    for SEED in "\${SEEDS[@]}"; do

      RUNID="\${MODEL_TAG}_\${TASK}_\${METHOD}_b\${BUDGET}_s\${SEED}"
      OUTDIR="${ROOT}/outputs/\${RUNID}"
      EVALJSON="${ROOT}/results/main/\${RUNID}_eval200.json"
      METAJSON="${ROOT}/results/main/\${RUNID}_meta.json"

      rm -rf "\${OUTDIR}"
      rm -f "\${EVALJSON}" "\${METAJSON}"

      if [ "\$TASK" = "gsm8k" ]; then
        TRAIN_FILE="${ROOT}/data/gsm8k/train_\${BUDGET}.jsonl"
        TEST_FILE="${ROOT}/data/gsm8k/test.jsonl"
      elif [ "\$TASK" = "dialogsum" ]; then
        TRAIN_FILE="${ROOT}/data/dialogsum/train_\${BUDGET}.jsonl"
        TEST_FILE="${ROOT}/data/dialogsum/test.jsonl"
      elif [ "\$TASK" = "squad_v2" ]; then
        TRAIN_FILE="${ROOT}/data/squad_v2/train_\${BUDGET}.jsonl"
        TEST_FILE="${ROOT}/data/squad_v2/validation.jsonl"
      else
        echo "Unknown task: \$TASK"
        exit 1
      fi

      echo "===== RUN START ====="
      echo "RUNID=\${RUNID}"
      echo "TASK=\${TASK}"
      echo "BUDGET=\${BUDGET}"
      echo "SEED=\${SEED}"
      echo "TIME=\$(date)"
      nvidia-smi

      python ${ROOT}/scripts/train/train_peft.py \
        --method "\${METHOD}" \
        --model_name_or_path "\${MODEL_NAME}" \
        --train_file "\${TRAIN_FILE}" \
        --output_dir "\${OUTDIR}" \
        --learning_rate "\${LR}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --max_length 1024 \
        --seed "\${SEED}"

      if [ "\$TASK" = "gsm8k" ]; then
        python ${ROOT}/scripts/eval/eval_gsm8k.py \
          --base_model "\${MODEL_NAME}" \
          --adapter_path "\${OUTDIR}" \
          --test_file "\${TEST_FILE}" \
          --output_file "\${EVALJSON}" \
          --max_examples 200 \
          --batch_size 4 \
          --max_new_tokens 32
      elif [ "\$TASK" = "dialogsum" ]; then
        python ${ROOT}/scripts/eval/eval_dialogsum.py \
          --base_model "\${MODEL_NAME}" \
          --adapter_path "\${OUTDIR}" \
          --test_file "\${TEST_FILE}" \
          --output_file "\${EVALJSON}" \
          --max_examples 200 \
          --batch_size 4 \
          --max_new_tokens 64
      elif [ "\$TASK" = "squad_v2" ]; then
        python ${ROOT}/scripts/eval/eval_squadv2.py \
          --base_model "\${MODEL_NAME}" \
          --adapter_path "\${OUTDIR}" \
          --test_file "\${TEST_FILE}" \
          --output_file "\${EVALJSON}" \
          --max_examples 200 \
          --batch_size 4 \
          --max_new_tokens 32
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

      echo "===== RUN END ====="
      echo "RUNID=\${RUNID} finished at \$(date)"

    done
  done
done

echo "All runs finished for ${JOB_NAME}"
EOF

  chmod +x "${FILE}"
  echo "generated: ${FILE}"
}

make_job "qwen"  "Qwen/Qwen2.5-3B-Instruct" "lora"  "2e-4"
make_job "qwen"  "Qwen/Qwen2.5-3B-Instruct" "qlora" "1e-4"
make_job "qwen"  "Qwen/Qwen2.5-3B-Instruct" "dora"  "1e-4"
make_job "qwen"  "Qwen/Qwen2.5-3B-Instruct" "ia3"   "1e-3"

make_job "gemma" "google/gemma-2-2b-it"     "lora"  "2e-4"
make_job "gemma" "google/gemma-2-2b-it"     "qlora" "1e-4"
make_job "gemma" "google/gemma-2-2b-it"     "dora"  "1e-4"
make_job "gemma" "google/gemma-2-2b-it"     "ia3"   "1e-3"

echo "Done."

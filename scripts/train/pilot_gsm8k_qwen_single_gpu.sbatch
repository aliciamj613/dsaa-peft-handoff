cat > /hpc2hdd/home/mliu954/projects/dsaa_peft/scripts/train/pilot_gsm8k_qwen_single_gpu.sbatch <<'BASH'
#!/bin/bash
#SBATCH -J peft_pilot_qwen_single
#SBATCH -p long_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o /hpc2hdd/home/mliu954/projects/dsaa_peft/logs/%x_%j.out
#SBATCH -e /hpc2hdd/home/mliu954/projects/dsaa_peft/logs/%x_%j.err

set -eo pipefail

module load anaconda3
module load cuda/12.8

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate peft_llm
set -u

cd /hpc2hdd/home/mliu954/projects/dsaa_peft

mkdir -p /hpc2hdd/home/mliu954/projects/dsaa_peft/outputs
mkdir -p /hpc2hdd/home/mliu954/projects/dsaa_peft/results/pilot
mkdir -p /hpc2hdd/home/mliu954/projects/dsaa_peft/logs

CONFIGS=(
"lora 1e-4 qwen_gsm8k_lora_b512_s42_lr1e4"
"lora 2e-4 qwen_gsm8k_lora_b512_s42_lr2e4"
"qlora 1e-4 qwen_gsm8k_qlora_b512_s42_lr1e4"
"qlora 2e-4 qwen_gsm8k_qlora_b512_s42_lr2e4"
"dora 1e-4 qwen_gsm8k_dora_b512_s42_lr1e4"
"dora 2e-4 qwen_gsm8k_dora_b512_s42_lr2e4"
"ia3 5e-4 qwen_gsm8k_ia3_b512_s42_lr5e4"
"ia3 1e-3 qwen_gsm8k_ia3_b512_s42_lr1e3"
)

echo "===== JOB INFO ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "NODELIST=$SLURM_JOB_NODELIST"
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

for CFG in "${CONFIGS[@]}"; do
  METHOD=$(echo "$CFG" | awk '{print $1}')
  LR=$(echo "$CFG" | awk '{print $2}')
  RUNID=$(echo "$CFG" | awk '{print $3}')

  OUTDIR=/hpc2hdd/home/mliu954/projects/dsaa_peft/outputs/${RUNID}
  EVALJSON=/hpc2hdd/home/mliu954/projects/dsaa_peft/results/pilot/${RUNID}_eval200.json
  METAJSON=/hpc2hdd/home/mliu954/projects/dsaa_peft/results/pilot/${RUNID}_meta.json

  rm -rf "${OUTDIR}"
  rm -f "${EVALJSON}" "${METAJSON}"

  echo "===== RUN START ====="
  echo "RUNID=$RUNID"
  echo "METHOD=$METHOD"
  echo "LR=$LR"
  echo "TIME=$(date)"
  nvidia-smi

  python /hpc2hdd/home/mliu954/projects/dsaa_peft/scripts/train/train_peft.py \
    --method "${METHOD}" \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --train_file /hpc2hdd/home/mliu954/projects/dsaa_peft/data/gsm8k/train_512.jsonl \
    --output_dir "${OUTDIR}" \
    --learning_rate "${LR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 1024 \
    --seed 42

  python /hpc2hdd/home/mliu954/projects/dsaa_peft/scripts/eval/eval_gsm8k.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter_path "${OUTDIR}" \
    --test_file /hpc2hdd/home/mliu954/projects/dsaa_peft/data/gsm8k/test.jsonl \
    --output_file "${EVALJSON}" \
    --max_examples 200 \
    --batch_size 4 \
    --max_new_tokens 32

  python - <<PY
import os, json

runid = "${RUNID}"
outdir = "${OUTDIR}"
evaljson = "${EVALJSON}"
metajson = "${METAJSON}"

data = {
    "run_id": runid,
    "method": "${METHOD}",
    "lr": "${LR}",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "task": "gsm8k",
    "budget": 512,
    "seed": 42,
}

train_summary_path = os.path.join(outdir, "train_summary.json")
if os.path.exists(train_summary_path):
    data["train_summary"] = json.load(open(train_summary_path))

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

  echo "===== RUN END ====="
  echo "RUNID=$RUNID finished at $(date)"
done

echo "All pilot runs finished."
BASH
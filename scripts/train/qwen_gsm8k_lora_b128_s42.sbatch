#!/bin/bash
#SBATCH -J qwen_gsm8k_lora_b128_s42
#SBATCH -p i64m1tga40u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o ~/projects/dsaa_peft/logs/%x_%j.out
#SBATCH -e ~/projects/dsaa_peft/logs/%x_%j.err

set -euo pipefail

module load anaconda3
conda activate peft_llm

cd ~/projects/dsaa_peft

echo "===== JOB INFO ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "START_TIME=$(date)"
nvidia-smi

python scripts/train/train_peft.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --train_file ~/projects/dsaa_peft/data/gsm8k/train_128.jsonl \
  --output_dir ~/projects/dsaa_peft/outputs/qwen_gsm8k_lora_b128_s42 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --seed 42

echo "END_TIME=$(date)"
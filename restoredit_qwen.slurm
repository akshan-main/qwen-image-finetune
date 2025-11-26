#!/bin/bash
#SBATCH --job-name=restoredit_qwen
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/krishna/akshan3/logs/restoredit_qwen_%j.out
#SBATCH --error=/gscratch/krishna/akshan3/logs/restoredit_qwen_%j.err

set -eo pipefail
set -x

# Activate env
cd /mmfs1/gscratch/krishna/akshan3-ql/qwen-image-finetune
source .envs/qwenres/bin/activate

export HF_HOME=/gscratch/krishna/akshan3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

cd src

# Optional: cache step (you can comment this out after first successful run)
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m qflux.main \
  --config ../configs/restoredit_qwen_fp16.yaml \
  --cache

# Training
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m qflux.main \
  --config ../configs/restoredit_qwen_fp16.yaml

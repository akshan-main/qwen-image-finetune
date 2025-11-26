#!/bin/bash
#SBATCH --job-name=restoredit_qwen_a100
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/krishna/akshan3/logs/restoredit_qwen_a100_%j.out
#SBATCH --error=/gscratch/krishna/akshan3/logs/restoredit_qwen_a100_%j.err

set -eo pipefail
set -x

# 1) Wire up micromamba in the batch shell
set +u; source ~/.bashrc; set -u
micromamba activate qwenres

# 2) Go to project
cd /mmfs1/gscratch/krishna/akshan3-ql/qwen-image-finetune

export HF_HOME=/gscratch/krishna/akshan3/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

cd src

# 3) Cache step (one-time; harmless if re-run)
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m qflux.main \
  --config ../configs/restoredit_qwen_fp16.yaml \
  --cache

# 4) Training
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m qflux.main \
  --config ../configs/restoredit_qwen_fp16.yaml

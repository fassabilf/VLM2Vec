#!/bin/bash
# ==============================================================
# Template: evaluate a local .pt checkpoint on MMEB v1
# without uploading to HuggingFace.
#
# Fill in the four variables below, then submit with:
#   sbatch scripts/sbatch_mmebv1_local_template.sh
# ==============================================================

# ---- Configure these four variables -------------------------
export ARCH="ViT-T-16"                    # open_clip architecture name
export WEIGHTS="/path/to/epoch_32.pt"     # absolute path to .pt checkpoint
export RUN_NAME="my-experiment"           # used as result folder name
export MODALITY="image"                   # "image" or "visdoc"
# -------------------------------------------------------------

#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A lt200394
#SBATCH -J mmebv1_local_${RUN_NAME}
#SBATCH -o ./logs/%x_%j.out

set -euo pipefail

export HF_HOME="/project/lt200394-thllmV/benchmark/.cache/huggingface"
export HF_DATASETS_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/datasets"
export HF_HUB_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load cuda/11.8
module load gcc/12.2.0

export TMPDIR=/project/lt200394-thllmV/benchmark/.tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

module load Mamba/23.11.0-0
conda activate vlm2vec_env

export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=64
export OUTPUT_PATH="/project/lt200394-thllmV/benchmark/VLM2Vec/results/${RUN_NAME}/${MODALITY}/"

cd /project/lt200394-thllmV/benchmark/VLM2Vec
bash ./scripts/run_mmebv1_local.sh

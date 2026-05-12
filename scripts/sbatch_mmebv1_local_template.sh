#!/bin/bash
# ==============================================================
# Evaluate a local .pt checkpoint on MMEB v1 (image + visdoc).
#
# Usage: just run it directly — it submits both modalities.
#   bash scripts/sbatch_mmebv1_local_template.sh
# ==============================================================

# ---- Configure these three variables ------------------------
ARCH="ViT-T-16"
WEIGHTS="/project/lt200394-thllmV/multilingual-clip-kd/open_clip/experiments/siglip2_kd/clipkd_ViT-T-16_from_ViT-B-16-SigLIP2_v2/checkpoints/epoch_latest.pt"
RUN_NAME="clipkd-ViT-T-16-from-SigLIP2-v2"
# -------------------------------------------------------------

# If not inside a SLURM job yet, submit self as array (1=image, 2=visdoc)
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    exec sbatch \
        --array="1-2" \
        --export="ARCH=${ARCH},WEIGHTS=${WEIGHTS},RUN_NAME=${RUN_NAME}" \
        -p gpu -N 1 -c 16 --gpus-per-node=1 --ntasks-per-node=1 \
        -t 12:00:00 -A lt200394 \
        -J "mmebv1_local_${RUN_NAME}" \
        -o "/project/lt200394-thllmV/benchmark/VLM2Vec/logs/%x_%A_%a.out" \
        "$0"
fi

# === Inside SLURM array job ===
# array index 1 = image, 2 = visdoc
MODALITIES=("" "image" "visdoc")
export MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

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

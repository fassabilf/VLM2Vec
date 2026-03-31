#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A lt200394
#SBATCH -J mmeb_clipkd-released-ViT-T-16-baseline-cc12m_visdoc
#SBATCH -o /project/lt200394-thllmV/benchmark/VLM2Vec/logs/%x_%j.out

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

echo "=== toolchain ==="
which gcc; gcc --version
which nvcc; nvcc --version
python --version || true

module load Mamba/23.11.0-0
conda activate vlm2vec_env

export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=64
export OUTPUT_PATH="/project/lt200394-thllmV/benchmark/VLM2Vec/results/clipkd-released-ViT-T-16-baseline-cc12m/visdoc/"
export DATA_BASEDIR="data/vlm2vec_eval"

cd "/project/lt200394-thllmV/benchmark/VLM2Vec"
bash "/project/lt200394-thllmV/benchmark/VLM2Vec/scripts/run_mmeb_clipkd-released-ViT-T-16-baseline-cc12m_visdoc.sh"

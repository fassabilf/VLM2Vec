#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_BASEDIR="${DATA_BASEDIR:-data/vlm2vec_eval}"
MODALITY="visdoc"
DATA_CONFIG_PATH="experiments/public/eval/${MODALITY}.yaml"

MODEL_NAME="apple/DFN2B-CLIP-ViT-B-16"
MODEL_BACKBONE="openclip"
OUTPUT_PATH="${OUTPUT_PATH:-./results/DFN2B-CLIP-ViT-B-16/${MODALITY}/}"

echo "================================================="
echo "🚀 Processing Model: ${MODEL_NAME}"
echo "  - Modality: ${MODALITY}"
echo "  - Output Path: ${OUTPUT_PATH}"
echo "================================================="

mkdir -p "${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 eval.py \
  --pooling eos \
  --normalize true \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --dataloader_num_workers 4 \
  --model_backbone "${MODEL_BACKBONE}" \
  --model_name "${MODEL_NAME}" \
  --model_type "${MODEL_BACKBONE}" \
  --dataset_config "${DATA_CONFIG_PATH}" \
  --encode_output_path "${OUTPUT_PATH}" \
  --data_basedir "${DATA_BASEDIR}" \
  --output_dir "${OUTPUT_PATH}"

echo "Done."

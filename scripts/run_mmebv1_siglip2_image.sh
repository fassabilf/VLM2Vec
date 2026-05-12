#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATA_BASEDIR="${DATA_BASEDIR:-data/vlm2vec_eval}"

MODALITY="image"
DATA_CONFIG_PATH="experiments/public/eval/${MODALITY}.yaml"

MODEL_NAME="google/siglip2-so400m-patch16-naflex"
MODEL_BACKBONE="siglip"

OUTPUT_PATH="/project/lt200394-thllmV/benchmark/VLM2Vec/results/SigLIP2-so400m-patch16-naflex/${MODALITY}/"

echo "================================================="
echo "🚀 Processing Model: ${MODEL_NAME}"
echo "  - Modality: ${MODALITY}"
echo "  - Output Path: ${OUTPUT_PATH}"
echo "================================================="

mkdir -p "${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python eval.py \
  --pooling eos \
  --normalize true \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --model_backbone "${MODEL_BACKBONE}" \
  --model_name "${MODEL_NAME}" \
  --dataset_config "${DATA_CONFIG_PATH}" \
  --encode_output_path "${OUTPUT_PATH}" \
  --data_basedir "${DATA_BASEDIR}"

echo "✅ Done."
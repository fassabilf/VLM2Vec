#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATA_BASEDIR="${DATA_BASEDIR:-data/vlm2vec_eval}"

MODALITY="visdoc"
DATA_CONFIG_PATH="experiments/public/eval/${MODALITY}.yaml"

MODEL_NAME="facebook/metaclip-2-worldwide-huge-quickgelu"
MODEL_BACKBONE="metaclip2"

OUTPUT_PATH="${OUTPUT_PATH:-./results/MetaCLIP2-worldwide-huge-quickgelu/${MODALITY}/}"

echo "================================================="
echo "Processing Model: ${MODEL_NAME}"
echo "  - Modality: ${MODALITY}"
echo "  - Output Path: ${OUTPUT_PATH}"
echo "================================================="

mkdir -p "${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 eval.py \
  --pooling eos \
  --normalize true \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --model_backbone "${MODEL_BACKBONE}" \
  --model_name "${MODEL_NAME}" \
  --model_type "${MODEL_BACKBONE}" \
  --dataset_config "${DATA_CONFIG_PATH}" \
  --encode_output_path "${OUTPUT_PATH}" \
  --data_basedir "${DATA_BASEDIR}" \
  --output_dir "${OUTPUT_PATH}"

echo "Done."

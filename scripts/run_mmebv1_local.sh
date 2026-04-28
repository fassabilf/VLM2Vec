#!/usr/bin/env bash
# Run MMEB eval from a local .pt checkpoint — no HuggingFace upload needed.
#
# Usage (set env vars before calling):
#   ARCH=ViT-T-16 WEIGHTS=/path/to/epoch_32.pt RUN_NAME=my-run MODALITY=image \
#   bash scripts/run_mmebv1_local.sh
#
# Or from sbatch_mmebv1_local_template.sh (sets the vars for you).
set -euo pipefail

ARCH="${ARCH:-ViT-T-16}"
WEIGHTS="${WEIGHTS:-/path/to/epoch_32.pt}"
RUN_NAME="${RUN_NAME:-local-run}"
MODALITY="${MODALITY:-image}"          # image or visdoc

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_BASEDIR="${DATA_BASEDIR:-data/vlm2vec_eval}"
DATA_CONFIG_PATH="experiments/public/eval/${MODALITY}.yaml"
OUTPUT_PATH="${OUTPUT_PATH:-./results/${RUN_NAME}/${MODALITY}/}"

echo "================================================="
echo "  arch     : ${ARCH}"
echo "  weights  : ${WEIGHTS}"
echo "  run-name : ${RUN_NAME}"
echo "  modality : ${MODALITY}"
echo "  output   : ${OUTPUT_PATH}"
echo "================================================="

mkdir -p "${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 eval.py \
  --pooling eos \
  --normalize true \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --dataloader_num_workers 4 \
  --model_backbone openclip \
  --model_name "${ARCH}" \
  --checkpoint_path "${WEIGHTS}" \
  --model_type openclip \
  --dataset_config "${DATA_CONFIG_PATH}" \
  --encode_output_path "${OUTPUT_PATH}" \
  --data_basedir "${DATA_BASEDIR}" \
  --output_dir "${OUTPUT_PATH}"

echo "Done."

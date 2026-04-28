#!/usr/bin/env bash
# =============================================================
# run_mmeb.sh — MMEB(v1) / VLM2Vec evaluation launcher
#
# Usage:
#   bash run_mmeb.sh --model 1 --modality image
#   bash run_mmeb.sh --model 1,2,3 --modality both
#   bash run_mmeb.sh --all --split 1
#   bash run_mmeb.sh --all --split 2
#   bash run_mmeb.sh --list
# =============================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────
VLM2VEC_DIR="/project/lt200394-thllmV/benchmark/VLM2Vec"
RESULTS_BASE="${VLM2VEC_DIR}/results"
SLURM_LOG_DIR="${VLM2VEC_DIR}/logs"
SCRIPTS_DIR="${VLM2VEC_DIR}/scripts"
DATA_BASEDIR="data/vlm2vec_eval"
HF_CACHE="/project/lt200394-thllmV/benchmark/.cache/huggingface"

BATCH_SIZE="${BATCH_SIZE:-64}"
MODALITIES=("image" "visdoc")

# ── Model Registry ─────────────────────────────────────────────
MODELS=(
    "1|fassabilf/clipkd-ViT-T-16-cc12m|Our ViT-T-16 KD (CC12M, 42.83% IN)"
    "2|fassabilf/clipkd-released-ViT-T-16-baseline-cc12m|CLIP-KD Released ViT-T-16 Baseline (30.55% IN)"
    "3|fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16teacher-cc12m|CLIP-KD Released ViT-T-16 KD ViT-B/16 teacher (34.90% IN)"
    "4|fassabilf/clipkd-released-ViT-T-16-clipkd-vitb16laion-cc12m|CLIP-KD Released ViT-T-16 KD ViT-B/16 Laion teacher (42.6% IN)"
    "5|fassabilf/clipkd-released-ViT-B-16-teacher-cc12m|CLIP-KD Released ViT-B/16 Teacher CC12M (36.99% IN)"
    "6|fassabilf/clipkd-released-ViT-B-16-teacher-laion400m|CLIP-KD Released ViT-B/16 Teacher Laion400M (67.1% IN)"
    "7|laion/CLIP-ViT-B-32-laion2B-s34B-b79K|LAION ViT-B-32 Teacher"
)

get_model_id()   { echo "$1" | cut -d'|' -f1; }
get_model_repo() { echo "$1" | cut -d'|' -f2; }
get_model_desc() { echo "$1" | cut -d'|' -f3; }
get_short_name() { echo "${1##*/}"; }

print_models() {
    echo ""
    echo "Available Models:"
    echo "────────────────────────────────────────────────────────────"
    for entry in "${MODELS[@]}"; do
        printf "  [%s] %s\n      %s\n\n" \
            "$(get_model_id "$entry")" \
            "$(get_model_repo "$entry")" \
            "$(get_model_desc "$entry")"
    done
    echo "────────────────────────────────────────────────────────────"
    echo "Modalities: image | visdoc | both"
}

resolve_model() {
    local input="$1"
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        for entry in "${MODELS[@]}"; do
            if [[ "$(get_model_id "$entry")" == "$input" ]]; then
                get_model_repo "$entry"; return
            fi
        done
        echo "ERROR: Model ID $input not found" >&2; exit 1
    fi
    echo "$input"
}

submit_job() {
    local model_repo="$1"
    local modality="$2"
    local short_name job_name run_script sbatch_script output_path
    short_name=$(get_short_name "$model_repo")
    job_name="mmeb_${short_name}_${modality}"
    run_script="${SCRIPTS_DIR}/run_mmeb_${short_name}_${modality}.sh"
    sbatch_script="${SCRIPTS_DIR}/sbatch_mmeb_${short_name}_${modality}.sh"
    output_path="${RESULTS_BASE}/${short_name}/${modality}/"

    mkdir -p "$SCRIPTS_DIR" "$SLURM_LOG_DIR"

    # ── Inner run script ──────────────────────────────────────
    cat > "$run_script" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="\${BATCH_SIZE:-${BATCH_SIZE}}"
DATA_BASEDIR="\${DATA_BASEDIR:-${DATA_BASEDIR}}"
MODALITY="${modality}"
DATA_CONFIG_PATH="experiments/public/eval/\${MODALITY}.yaml"

MODEL_NAME="${model_repo}"
MODEL_BACKBONE="openclip"
OUTPUT_PATH="\${OUTPUT_PATH:-${output_path}}"

echo "================================================="
echo "Model   : \${MODEL_NAME}"
echo "Modality: \${MODALITY}"
echo "Output  : \${OUTPUT_PATH}"
echo "================================================="

mkdir -p "\${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES}" python3 eval.py \\
  --pooling eos \\
  --normalize true \\
  --per_device_eval_batch_size "\${BATCH_SIZE}" \\
  --dataloader_num_workers 4 \\
  --model_backbone "\${MODEL_BACKBONE}" \\
  --model_name "\${MODEL_NAME}" \\
  --model_type "\${MODEL_BACKBONE}" \\
  --dataset_config "\${DATA_CONFIG_PATH}" \\
  --encode_output_path "\${OUTPUT_PATH}" \\
  --data_basedir "\${DATA_BASEDIR}" \\
  --output_dir "\${OUTPUT_PATH}"

echo "Done."
RUNEOF

    # ── Slurm job ─────────────────────────────────────────────
    cat > "$sbatch_script" <<SBATCHEOF
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A lt200394
#SBATCH -J ${job_name}
#SBATCH -o ${SLURM_LOG_DIR}/%x_%j.out

set -euo pipefail

export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load cuda/11.8
module load gcc/12.2.0

export TMPDIR=/project/lt200394-thllmV/benchmark/.tmp/\${SLURM_JOB_ID}
mkdir -p "\$TMPDIR"

export CC=\$(which gcc)
export CXX=\$(which g++)
export CUDAHOSTCXX=\$(which g++)
export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

echo "=== toolchain ==="
which gcc; gcc --version
which nvcc; nvcc --version
python --version || true

module load Mamba/23.11.0-0
conda activate vlm2vec_env

export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=${BATCH_SIZE}
export OUTPUT_PATH="${output_path}"
export DATA_BASEDIR="${DATA_BASEDIR}"

cd "${VLM2VEC_DIR}"
bash "${run_script}"
SBATCHEOF

    echo "  → Submitting: ${job_name}"
    sbatch "$sbatch_script"
}

# ── Argument parsing ───────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    print_models
    echo ""
    echo "Usage: bash run_mmeb.sh --model <id_or_repo> --modality <image|visdoc|both>"
    echo "       bash run_mmeb.sh --all --modality both"
    echo "       bash run_mmeb.sh --all --split 1"
    echo "       bash run_mmeb.sh --all --split 2"
    exit 0
fi

SELECTED=()
SELECTED_MODALITY="both"
SPLIT=""
MODE="manual"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)     print_models; exit 0 ;;
        --all)      MODE="all"; shift ;;
        --split)    SPLIT="$2"; shift 2 ;;
        --modality) SELECTED_MODALITY="$2"; shift 2 ;;
        --model)
            IFS=',' read -ra SELECTED <<< "$2"
            shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "all" ]]; then
    for entry in "${MODELS[@]}"; do
        SELECTED+=("$(get_model_id "$entry")")
    done
fi

# ── Build job list ─────────────────────────────────────────────
ALL_JOBS=()
for sel in "${SELECTED[@]}"; do
    repo=$(resolve_model "$sel")
    if [[ "$SELECTED_MODALITY" == "both" ]]; then
        for mod in "${MODALITIES[@]}"; do
            ALL_JOBS+=("${repo}|${mod}")
        done
    else
        ALL_JOBS+=("${repo}|${SELECTED_MODALITY}")
    fi
done

# ── Apply split ────────────────────────────────────────────────
TOTAL=${#ALL_JOBS[@]}
if [[ -n "$SPLIT" && $TOTAL -gt 0 ]]; then
    MID=$(( (TOTAL + 1) / 2 ))
    if [[ "$SPLIT" == "1" ]]; then
        ALL_JOBS=("${ALL_JOBS[@]:0:$MID}")
        echo "Running Part 1 of 2 (jobs 1–${MID} of ${TOTAL})"
    else
        ALL_JOBS=("${ALL_JOBS[@]:$MID}")
        echo "Running Part 2 of 2 (jobs $((MID+1))–${TOTAL} of ${TOTAL})"
    fi
fi

# ── Submit ─────────────────────────────────────────────────────
echo ""
echo "Submitting ${#ALL_JOBS[@]} MMEB job(s)..."
echo "────────────────────────────────────────────────────────────"
for job in "${ALL_JOBS[@]}"; do
    repo="${job%%|*}"
    modality="${job##*|}"
    submit_job "$repo" "$modality"
done
echo "────────────────────────────────────────────────────────────"
echo "Done! Check: squeue -u \$USER"
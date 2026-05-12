# Evaluating a Local Checkpoint on MMEB

Run MMEB v1 eval directly from a local `.pt` checkpoint — no HuggingFace upload needed.

## Files

| File | Description |
|------|-------------|
| `scripts/run_mmebv1_local.sh` | Parameterized runner script |
| `scripts/sbatch_mmebv1_local_template.sh` | Slurm job template for HPC (ThaiSC) |

---

## Quick Start

### 1. Set variables and run

```bash
cd /path/to/VLM2Vec

ARCH=ViT-T-16 \
WEIGHTS=/path/to/epoch_32.pt \
RUN_NAME=siglip2-gemma-cc12m-ep32 \
MODALITY=image \
bash scripts/run_mmebv1_local.sh
```

### 2. Visual document tasks

```bash
ARCH=ViT-T-16 \
WEIGHTS=/path/to/epoch_32.pt \
RUN_NAME=siglip2-gemma-cc12m-ep32 \
MODALITY=visdoc \
bash scripts/run_mmebv1_local.sh
```

---

## Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCH` | `ViT-T-16` | open_clip architecture name, e.g. `ViT-T-16`, `ViT-B-16` |
| `WEIGHTS` | `/path/to/epoch_32.pt` | Absolute path to `.pt` checkpoint file |
| `RUN_NAME` | `local-run` | Name used for the result output folder |
| `MODALITY` | `image` | `image` or `visdoc` |
| `BATCH_SIZE` | `64` | Per-device eval batch size |
| `DATA_BASEDIR` | `data/vlm2vec_eval` | Root directory of MMEB dataset |

---

## Submitting on HPC (ThaiSC)

1. Edit the four variables at the top of `scripts/sbatch_mmebv1_local_template.sh`:

```bash
export ARCH="ViT-T-16"
export WEIGHTS="/path/to/epoch_32.pt"
export RUN_NAME="siglip2-gemma-cc12m-ep32"
export MODALITY="image"
```

2. Submit:

```bash
sbatch scripts/sbatch_mmebv1_local_template.sh
```

Logs go to `./logs/`.  
Results go to `/project/lt200394-thllmV/benchmark/VLM2Vec/results/<RUN_NAME>/<MODALITY>/`.

---

## Results

Scores are printed per-dataset at the end of the run and saved as JSON under the output path:

```
results/<RUN_NAME>/<MODALITY>/
├── ImageNet-1K_scores.json
├── ...
└── average_scores.json
```

---

## How It Works

`openclip_inference.py` detects whether `checkpoint_path` is a local file or a HuggingFace repo:

- **Local `.pt`** → `open_clip.create_model_and_transforms(arch, pretrained=path)`
- **HF Hub** → `open_clip.create_model_from_pretrained("hf-hub:org/repo")` *(existing behavior, unchanged)*

The `checkpoint_path` is passed through `ModelArguments` → `MMEBModel.load()` → `OpenCLIPModel(model_path=...)`.

# MMEB Evaluation: Bug Fix & Optimization Report

**Branch:** `metaclip2-eval-results`
**Date:** 2026-04-28
**Author:** Faiz Assabil Firdaus

---

## Overview

During the preparation of MMEB image evaluation for MetaCLIP2, two issues were identified and resolved:

1. **A correctness bug** — task instruction text was being silently injected into image embeddings for all CLIP-style backbones (OpenCLIP, SigLIP, MetaCLIP2), corrupting embeddings across all 36 MMEB tasks.
2. **A performance bottleneck** — serial image preprocessing in the main thread was leaving the GPU idle more than 95% of the time between batches.

Both issues have been fixed. The MetaCLIP2 results reported in this repository reflect the corrected evaluation pipeline.

---

## Part 1 — Bug Fix: Instruction Text Injection in CLIP-Style Backbones

**Commit:** `a61808e`
**Files changed:** `src/model/processor.py`, `src/data/eval_dataset/image_t2i_eval.py`

### 1.1 Background

The VLM2Vec evaluation framework uses a `process_input_text()` function to prepare text inputs before encoding. For instruction-following VLMs (LLaVA, Phi-3.5, etc.), this function wraps the raw text with a task-specific instruction (e.g., `"Represent the given image for classification: <image>"`). CLIP-style dual-encoder models — OpenCLIP, SigLIP, and MetaCLIP2 — do **not** benefit from instruction wrapping: they were pre-trained on raw image–caption pairs, so retrieval instruction strings are out-of-distribution for their text encoders.

### 1.2 Root Cause

`process_input_text()` had three near-identical branches for OPENCLIP, SIGLIP, and METACLIP2, each of which returned `instruction + " "` when `text` was empty:

```python
# Before — same pattern repeated for SIGLIP, METACLIP2, OPENCLIP:
elif model_backbone == SIGLIP:
    if text:
        return instruction + " " + text
    else:
        return instruction + " "      # ← always non-empty!
```

Because this always returned a non-empty string, the downstream function `get_fused_embeddings()` interpreted it as `has_text=True`. The instruction text was then **encoded by the text encoder and added to the image embedding** for every image-only input across all 36 MMEB tasks.

Concretely, this affected:
- Classification queries (image only, no text)
- Image-to-text (i2t) query images
- Text-to-image (t2i) candidate images
- Image-to-image (i2i) candidate images

Every image embedding for every CLIP-style baseline was contaminated with the instruction text representation.

### 1.3 Secondary Bug in `image_t2i_eval.py`

A second issue was found in `src/data/eval_dataset/image_t2i_eval.py`. The `data_prepare()` function constructed the t2i query text by directly concatenating `qry_inst`, bypassing `process_input_text()` entirely:

```python
# Before:
qry_inst = qry_inst.replace("<|image_1|>", VLM_IMAGE_TOKENS[model_backbone])
query_text = qry_inst + ' ' + qry_text + '\n'
```

This meant the instruction prefix was **always** applied to t2i query texts for CLIP-style models, regardless of the backbone check.

### 1.4 Fix

**`src/model/processor.py`** — The three redundant branches were consolidated into one that returns the raw text only, and returns `""` when text is empty so `has_text` stays `False`:

```python
# After:
elif model_backbone in [SIGLIP, METACLIP2, OPENCLIP]:
    # Dual-encoder CLIP-style models: return raw text only, no instruction wrapper.
    # Instructions are out-of-distribution for CLIP text encoders (trained on natural
    # captions). When text is empty the image side must be encoded pure — returning ""
    # keeps has_text=False in get_fused_embeddings so no instruction gets summed in.
    return text.strip() if text and text.strip() else ""
```

**`src/data/eval_dataset/image_t2i_eval.py`** — Added a backbone-conditional path for CLIP-style models:

```python
# After:
if model_backbone in [OPENCLIP, SIGLIP, METACLIP2]:
    # CLIP-style models: use raw query text only, no instruction wrapper
    query_text = qry_text.strip() if qry_text and qry_text.strip() else qry_inst
else:
    qry_inst = qry_inst.replace("<|image_1|>", VLM_IMAGE_TOKENS[model_backbone])
    query_text = qry_inst + ' ' + qry_text + '\n'
```

### 1.5 Impact

| Dimension | Detail |
|-----------|--------|
| Affected models | All CLIP-style baselines: OpenCLIP, SigLIP, MetaCLIP2 |
| Affected tasks | All 36 MMEB image tasks |
| Nature of error | Image embeddings were mixed with instruction-text semantics, degrading visual retrieval quality |
| Scope of fix | All image-only encoding paths (query and candidate side) |

---

## Part 2 — Performance Fix: GPU Underutilization

**Commit:** `f95304d`
**Files changed:** `src/model/baseline_backbone/openclip/openclip_inference.py`, `src/model/baseline_backbone/siglip/siglip_inference.py`, `eval.py`, all `scripts/run_mmebv1_*.sh`

### 2.1 Root Cause

`_encode_images()` in `openclip_inference.py` was preprocessing images serially inside a list comprehension in the main Python thread:

```python
# Before:
image_tensor = torch.stack([self.preprocess(img) for img in pil_images]).to(
    self.device
)
```

PIL image decoding and resizing operations are CPU-bound. When executed serially in the main thread, each operation must complete before the next begins, and the GPU sits idle waiting for a full batch of preprocessed tensors. In practice, GPU utilization dropped to below 5% between forward passes.

### 2.2 Changes Made

#### `openclip_inference.py` — Parallel preprocessing

PIL operations release the Python GIL when executed in threads, enabling genuine CPU parallelism via `ThreadPoolExecutor`:

```python
# After:
_NUM_WORKERS = 4

# Parallelize preprocessing across worker threads (PIL ops release the GIL)
with concurrent.futures.ThreadPoolExecutor(max_workers=_NUM_WORKERS) as ex:
    tensors = list(ex.map(self.preprocess, pil_images))
image_tensor = torch.stack(tensors).to(self.device, non_blocking=True)
```

`non_blocking=True` on `.to(device)` allows the CPU→GPU DMA transfer to overlap with subsequent CPU work, rather than blocking until the copy is complete.

#### `siglip_inference.py` — Non-blocking transfers

```python
# Before:
inputs = {k: v.to(self.device) for k, v in inputs.items()}

# After:
inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
```

Applied to both `_encode_text` and `_encode_images`.

#### `eval.py` — Pinned memory DataLoaders

```python
# Before:
eval_qry_loader = DataLoader(..., num_workers=training_args.dataloader_num_workers)

# After:
eval_qry_loader = DataLoader(..., num_workers=training_args.dataloader_num_workers,
                             pin_memory=torch.cuda.is_available())
```

`pin_memory=True` allocates batch tensors in page-locked (pinned) host memory. The CUDA DMA engine can then transfer tensors directly to GPU VRAM without an intermediate copy through pageable memory, reducing transfer latency.

#### All eval scripts — DataLoader workers

```bash
# Added to all scripts/run_mmebv1_*.sh and run_mmeb.sh:
--dataloader_num_workers 4
```

With `num_workers=4`, the DataLoader spawns 4 worker processes to prefetch and decode the next batch in parallel while the GPU is processing the current batch, eliminating the decode-wait cycle between batches.

### 2.3 Why Each Fix Works

| Fix | Mechanism |
|-----|-----------|
| `ThreadPoolExecutor` in `_encode_images` | PIL ops release the GIL → 4 threads give true CPU parallelism for decode + resize |
| `non_blocking=True` on `.to(device)` | CPU→GPU DMA overlaps with other CPU work; GPU pipeline is not stalled waiting for data |
| `pin_memory=True` in DataLoader | Tensors in pinned memory are DMA-transferred directly to VRAM, bypassing the pageable-memory intermediate buffer |
| `--dataloader_num_workers 4` | Next batch is prefetched and decoded while GPU processes the current batch |

Together, these changes eliminate the main CPU-side bottlenecks that were causing the GPU to idle between batches.

---

## Summary

| Issue | Commit | Root Cause | Fix |
|-------|--------|-----------|-----|
| Image embedding corruption | `a61808e` | `process_input_text()` always returned non-empty string for CLIP backbones → `has_text=True` → instruction text added to image embeddings | Return `""` when text is empty; return raw text only (no instruction) for CLIP-style models |
| t2i query instruction bypass | `a61808e` | `image_t2i_eval.py` concatenated `qry_inst` directly without backbone check | Added backbone-conditional path; CLIP uses raw `qry_text` only |
| GPU underutilization | `f95304d` | Serial PIL preprocessing in main thread; blocking `.to(device)`; no DataLoader prefetch | `ThreadPoolExecutor`, `non_blocking=True`, `pin_memory=True`, `num_workers=4` |

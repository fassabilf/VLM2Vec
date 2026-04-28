"""
Download everything needed to run MetaCLIP2 MMEB image eval locally.

Downloads to:
  - Model  : /workspace/.hf_home/hub/
  - Dataset: /workspace/.hf_home/datasets/
  - Images : /workspace/VLM2Vec/data/vlm2vec_eval/image-tasks/
"""
import os, sys

HF_HOME          = "/workspace/.hf_home"
HF_HUB_CACHE     = f"{HF_HOME}/hub"
HF_DATASETS_CACHE= f"{HF_HOME}/datasets"
DATA_BASEDIR     = "/workspace/VLM2Vec/data/vlm2vec_eval"
IMAGE_TASKS_DIR  = f"{DATA_BASEDIR}/image-tasks"

os.environ["HF_HOME"]           = HF_HOME
os.environ["HF_HUB_CACHE"]      = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HUB_OFFLINE"]    = "0"
os.environ["HF_DATASETS_OFFLINE"]= "0"

os.makedirs(HF_HUB_CACHE, exist_ok=True)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(IMAGE_TASKS_DIR, exist_ok=True)

from huggingface_hub import snapshot_download
from datasets import load_dataset, get_dataset_config_names

# ─── 1. MetaCLIP2 model ───────────────────────────────────────────────────────
MODEL_REPO = "facebook/metaclip-2-worldwide-huge-quickgelu"
print(f"\n{'='*60}")
print(f"Downloading model: {MODEL_REPO}")
print(f"{'='*60}")
snapshot_download(
    repo_id=MODEL_REPO,
    repo_type="model",
    cache_dir=HF_HUB_CACHE,
)
print("✓ Model downloaded")

# ─── 2. MMEB_Test_Instruct metadata (all image task configs) ─────────────────
MMEB_INSTRUCT_REPO = "ziyjiang/MMEB_Test_Instruct"
IMAGE_TASK_CONFIGS = [
    # Classification
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397",
    "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
    # QA
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
    "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
    # i2t
    "MSCOCO_i2t", "VisualNews_i2t",
    # t2i
    "VisDial", "MSCOCO_t2i", "VisualNews_t2i", "WebQA", "EDIS", "Wiki-SS-NQ",
    # i2i / VG
    "CIRR", "NIGHTS", "OVEN", "FashionIQ", "MSCOCO", "RefCOCO",
    "RefCOCO-Matching", "Visual7W-Pointing",
]

print(f"\n{'='*60}")
print(f"Downloading MMEB metadata: {MMEB_INSTRUCT_REPO}")
print(f"{'='*60}")
for cfg in IMAGE_TASK_CONFIGS:
    try:
        ds = load_dataset(MMEB_INSTRUCT_REPO, cfg, split="test",
                          cache_dir=HF_DATASETS_CACHE)
        print(f"  ✓ {cfg} ({len(ds)} rows)")
    except Exception as e:
        print(f"  ✗ {cfg}: {e}")

# ─── 3. MMEB image files (TIGER-Lab/MMEB-V2) ─────────────────────────────────
print(f"\n{'='*60}")
print(f"Downloading MMEB image files → {IMAGE_TASKS_DIR}")
print("(this may be large — ~10-50 GB depending on tasks)")
print(f"{'='*60}")
snapshot_download(
    repo_id="TIGER-Lab/MMEB-V2",
    repo_type="dataset",
    cache_dir=HF_DATASETS_CACHE,
    local_dir=IMAGE_TASKS_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=["image-tasks/*"],
    ignore_patterns=["video-tasks/*", "visdoc-tasks/*"],
)
print("✓ Image files downloaded")

print("\n✓ All done! Run eval with:")
print("  HF_HOME=/workspace/.hf_home \\")
print("  DATA_BASEDIR=/workspace/VLM2Vec/data/vlm2vec_eval \\")
print("  OUTPUT_PATH=/workspace/VLM2Vec/results/MetaCLIP2/image/ \\")
print("  bash scripts/run_mmebv1_metaclip2_image.sh")

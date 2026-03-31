import os
import subprocess

# ==============================================================================
# Configuration
# ==============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)

CACHE_DIR = "/project/lt200394-thllmV/benchmark/.cache"
HF_HOME = CACHE_DIR
HF_HUB_CACHE = os.path.join(CACHE_DIR, "hub")
HF_DATASETS_CACHE = os.path.join(CACHE_DIR, "datasets")

# Set environment variables
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_HUB_CACHE

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    print(f"✅ HF Token loaded: hf_...{HF_TOKEN[-4:]}")
else:
    print("⚠️  No HF_TOKEN found. Set it via: export HF_TOKEN=hf_xxx")
    print("   Some gated repos may fail to download.")

# ==============================================================================
# Login to HuggingFace
# ==============================================================================
from huggingface_hub import login

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace Hub")

# ==============================================================================
# 1. Download Models
# ==============================================================================
from transformers import AutoConfig, AutoModel, AutoProcessor

MODELS = [
    # "Qwen/Qwen3-VL-Embedding-2B",
    # "google/siglip2-so400m-patch16-naflex",
]

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"📥 Downloading model: {model_name}")
    print(f"{'='*60}")
    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=HF_HUB_CACHE
        )
        print(f"   model_type: {config.model_type}")

        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=HF_HUB_CACHE
        )
        print(f"   ✅ Processor downloaded")

        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=HF_HUB_CACHE
        )
        print(f"   ✅ Model downloaded")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# ==============================================================================
# 2. Pull LFS data for MMEB-V2
# ==============================================================================
EVAL_DATA_DIR = "/project/lt200394-thllmV/benchmark/VLM2Vec/data/vlm2vec_eval"

if os.path.isdir(EVAL_DATA_DIR):
    print(f"\n{'='*60}")
    print(f"📥 Pulling LFS files in: {EVAL_DATA_DIR}")
    print(f"{'='*60}")

    # Configure git credential for LFS
    if HF_TOKEN:
        subprocess.run(
            ["git", "config", "credential.helper", "store"],
            cwd=EVAL_DATA_DIR,
        )
        # Write credential so git-lfs can authenticate
        credential_file = os.path.expanduser("~/.git-credentials")
        with open(credential_file, "a") as f:
            f.write(f"https://user:{HF_TOKEN}@huggingface.co\n")
        print("   ✅ Git credential configured for LFS")

    subprocess.run(["git", "lfs", "install"], cwd=EVAL_DATA_DIR)
    result = subprocess.run(["git", "lfs", "pull"], cwd=EVAL_DATA_DIR)

    if result.returncode == 0:
        print("   ✅ LFS pull complete")
    else:
        print("   ❌ LFS pull failed. Trying selective pull...")
        for folder in ["image-tasks", "video-tasks", "visdoc-tasks"]:
            print(f"   📥 Pulling {folder}...")
            subprocess.run(
                ["git", "lfs", "pull", f"--include={folder}/*"],
                cwd=EVAL_DATA_DIR,
            )
else:
    print(f"\n⚠️  {EVAL_DATA_DIR} not found. Clone the repo first:")
    print(f"   cd /project/lt200394-thllmV/benchmark/VLM2Vec/data/vlm2vec_eval")
    print(f"   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-V2 .")

# ==============================================================================
# 3. Verify
# ==============================================================================
print(f"\n{'='*60}")
print("🔍 Verification")
print(f"{'='*60}")

# Check models in cache
for model_name in MODELS:
    safe_name = model_name.replace("/", "--")
    model_path = os.path.join(HF_HUB_CACHE, f"models--{safe_name}")
    if os.path.isdir(model_path):
        print(f"   ✅ {model_name} cached")
    else:
        print(f"   ❌ {model_name} NOT cached")

# Check LFS files
if os.path.isdir(EVAL_DATA_DIR):
    test_file = os.path.join(EVAL_DATA_DIR, "image-tasks", "mmeb_v1.tar.gz")
    if os.path.isfile(test_file):
        size_mb = os.path.getsize(test_file) / (1024 * 1024)
        if size_mb > 1:
            print(f"   ✅ mmeb_v1.tar.gz = {size_mb:.0f} MB (real file)")
        else:
            print(f"   ❌ mmeb_v1.tar.gz = {size_mb:.2f} MB (probably LFS pointer!)")

print("\n🎉 Done!")
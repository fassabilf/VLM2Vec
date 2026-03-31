import os
import yaml
from typing import Optional, Tuple, List

from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import login

# ==============================================================================
# 0) Cache location
# ==============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)

CACHE_DIR = "/project/lt200394-thllmV/benchmark/.cache"
HF_HOME = CACHE_DIR
HF_HUB_CACHE = os.path.join(CACHE_DIR, "hub")
HF_DATASETS_CACHE = os.path.join(CACHE_DIR, "datasets")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_HUB_CACHE

# Ensure ONLINE while caching
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    login(token=HF_TOKEN)
    print(f"✅ Logged in (hf_...{HF_TOKEN[-4:]})")
else:
    print("⚠️ No HF_TOKEN (gated repos may fail).")

# ==============================================================================
# 1) Import your mapping
# ==============================================================================
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH

MMEB_TEST_INSTRUCT_REPO = "ziyjiang/MMEB_Test_Instruct"

# ==============================================================================
# 2) YAML discovery
# ==============================================================================
YAML_DIR = "/project/lt200394-thllmV/benchmark/VLM2Vec/experiments/public/eval"
YAML_FILES = ["visdoc.yaml"]  # auto cache all; edit if needed

# ==============================================================================
# 3) Helpers
# ==============================================================================
BEIR_CONFIGS_PRIORITY = ["corpus", "docs", "queries", "qrels"]

def load_yaml_tasks(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    out = []
    for task_key, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        ds_name = cfg.get("dataset_name")
        ds_parser = cfg.get("dataset_parser")
        if ds_name:
            out.append((task_key, ds_parser, ds_name))
    return out

def touch_dataset(ds):
    # Force download/materialization by accessing first item of each split
    if hasattr(ds, "keys"):  # DatasetDict
        for split in ds.keys():
            if len(ds[split]) > 0:
                _ = ds[split][0]
    else:  # Dataset
        if len(ds) > 0:
            _ = ds[0]

def cache_all_configs(repo_id: str):
    """
    Cache all configs for a dataset repo, BEIR-style or otherwise.
    """
    configs = get_dataset_config_names(repo_id)
    configs_sorted = (
        [c for c in BEIR_CONFIGS_PRIORITY if c in configs]
        + [c for c in configs if c not in BEIR_CONFIGS_PRIORITY]
    )
    print(f"   configs={configs_sorted}")

    for cfg in configs_sorted:
        print(f"   📥 load_dataset({repo_id!r}, {cfg!r})")
        ds = load_dataset(
            repo_id,
            cfg,
            cache_dir=HF_DATASETS_CACHE,
            download_mode="reuse_dataset_if_exists",
        )
        touch_dataset(ds)

def cache_one(repo_id: str, config_name: Optional[str], split: Optional[str]):
    """
    Cache a single (repo, config, split) if possible.
    If config is None: tries load_dataset(repo, split=...).
    """
    kwargs = dict(cache_dir=HF_DATASETS_CACHE, download_mode="reuse_dataset_if_exists")
    if config_name is None:
        ds = load_dataset(repo_id, split=split or "test", **kwargs)
    else:
        ds = load_dataset(repo_id, config_name, split=split or "test", **kwargs)
    touch_dataset(ds)
    # return length info (best effort)
    try:
        return len(ds)
    except Exception:
        return None

def should_fallback_to_all_configs(err: Exception) -> bool:
    msg = str(err)
    needles = [
        "Config name is missing",
        "BuilderConfig",
        "Please pick one among the available configs",
        "Available configs",
    ]
    return any(n in msg for n in needles)

def resolve_from_mapping(ds_name: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """
    Convert your EVAL_DATASET_HF_PATH entry to (repo_id, config, split).
    Note: In your mapping, the middle field is sometimes "lang" not HF config.
    We try it as config first, but will auto-fallback to cache_all_configs on mismatch.
    """
    if ds_name not in EVAL_DATASET_HF_PATH:
        return None
    repo_id, subset_or_lang, split = EVAL_DATASET_HF_PATH[ds_name]
    config = subset_or_lang if subset_or_lang not in (None, "", "None") else None
    return repo_id, config, split

# ==============================================================================
# 4) Load all YAMLs and cache
# ==============================================================================
tasks = []
for fname in YAML_FILES:
    path = os.path.join(YAML_DIR, fname)
    if os.path.isfile(path):
        tasks.extend(load_yaml_tasks(path))
    else:
        print(f"⚠️ Missing YAML: {path}")

unique_dataset_names = sorted(set(ds_name for _, _, ds_name in tasks))
print(f"\nFound {len(unique_dataset_names)} unique dataset_name entries from YAML.")

failed: List[tuple] = []

for ds_name in unique_dataset_names:
    print(f"\n{'='*80}\n📥 Caching dataset_name={ds_name}\n{'='*80}")

    mapped = resolve_from_mapping(ds_name)

    # A) Try mapped repo/config/split if available
    if mapped is not None:
        repo_id, config, split = mapped
        try:
            n = cache_one(repo_id, config, split)
            print(f"✅ Cached via mapping: repo={repo_id} config={config} split={split} (n={n})")
            continue
        except Exception as e:
            print(f"⚠️ Mapping path failed: repo={repo_id} config={config} split={split}\n   err={e}")

            # If config mismatch / missing config, cache ALL configs for that repo
            if should_fallback_to_all_configs(e):
                try:
                    print(f"🔁 Falling back to cache ALL configs for repo={repo_id}")
                    cache_all_configs(repo_id)
                    print(f"✅ Cached all configs for repo={repo_id}")
                    continue
                except Exception as e2:
                    print(f"❌ Failed caching all configs for repo={repo_id}: {e2}")
                    failed.append((ds_name, repo_id, config, split, f"all_configs_fail: {e2}"))
                    continue
            else:
                failed.append((ds_name, repo_id, config, split, f"mapping_fail: {e}"))
                continue

    # B) Fallback: treat ds_name as config under MMEB_Test_Instruct
    try:
        n = cache_one(MMEB_TEST_INSTRUCT_REPO, ds_name, "test")
        print(f"✅ Cached via fallback repo: repo={MMEB_TEST_INSTRUCT_REPO} config={ds_name} split=test (n={n})")
    except Exception as e:
        print(f"⚠️ Fallback config failed: {e}")

        # If config missing, cache all configs of the fallback repo
        if should_fallback_to_all_configs(e):
            try:
                print(f"🔁 Falling back to cache ALL configs for repo={MMEB_TEST_INSTRUCT_REPO}")
                cache_all_configs(MMEB_TEST_INSTRUCT_REPO)
                print(f"✅ Cached all configs for repo={MMEB_TEST_INSTRUCT_REPO}")
            except Exception as e2:
                print(f"❌ Failed caching all configs for fallback repo: {e2}")
                failed.append((ds_name, MMEB_TEST_INSTRUCT_REPO, ds_name, "test", f"fallback_all_configs_fail: {e2}"))
        else:
            failed.append((ds_name, MMEB_TEST_INSTRUCT_REPO, ds_name, "test", f"fallback_fail: {e}"))

# ==============================================================================
# 5) Summary
# ==============================================================================
print("\n\n==================== SUMMARY ====================")
if failed:
    print(f"⚠️ {len(failed)} datasets failed to cache:")
    for item in failed:
        print(" -", item)
else:
    print("✅ All datasets cached successfully!")

print(f"\nCache dirs:\n - HF_DATASETS_CACHE={HF_DATASETS_CACHE}\n - HF_HUB_CACHE={HF_HUB_CACHE}")
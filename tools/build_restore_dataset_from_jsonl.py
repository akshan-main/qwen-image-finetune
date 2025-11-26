import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence
from PIL import Image as PILImage

BASE = Path("/gscratch/krishna/akshan3")
RESTO_ROOT = BASE / "resto_pipeline"

TRAIN_JSON = RESTO_ROOT / "train.jsonl"
VAL_JSON = RESTO_ROOT / "test.jsonl"

MASK_PATH = RESTO_ROOT / "dummy_mask.png"

if not MASK_PATH.exists():
    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    m = PILImage.new("L", (512, 512), color=0)
    m.save(MASK_PATH)

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except:
                continue
    return rows

def pick_key(row, candidates):
    for k in candidates:
        if k in row:
            return k
    return None

def resolve_path(p: str) -> str:
    p = str(p).strip()
    if os.path.isabs(p):
        return p
    return str((RESTO_ROOT / p).resolve())

def build_split(rows):
    ids = []
    control_images = []
    control_mask = []
    target_image = []
    prompts = []

    for i, row in enumerate(rows):
        src_key = pick_key(row, ["source", "src", "input", "before"])
        tgt_key = pick_key(row, ["target", "tgt", "output", "after"])
        if src_key is None or tgt_key is None:
            continue

        try:
            src_path = resolve_path(row[src_key])
            tgt_path = resolve_path(row[tgt_key])
        except Exception:
            continue

        prompt = row.get("instruction", row.get("prompt", "restore this image"))

        ids.append(str(i))
        control_images.append([src_path])
        control_mask.append(str(MASK_PATH))
        target_image.append(tgt_path)
        prompts.append(prompt)

    data = {
        "id": ids,
        "control_images": control_images,
        "control_mask": control_mask,
        "target_image": target_image,
        "prompt": prompts,
    }
    return data

train_rows = load_jsonl(TRAIN_JSON)
val_rows = load_jsonl(VAL_JSON)

train_dict = build_split(train_rows)
val_dict = build_split(val_rows)

features = Features(
    {
        "id": Value("string"),
        "control_images": Sequence(Image()),
        "control_mask": Image(),
        "target_image": Image(),
        "prompt": Value("string"),
    }
)

train_ds = Dataset.from_dict(train_dict).cast(features)
val_ds = Dataset.from_dict(val_dict).cast(features)

ddict = DatasetDict({"train": train_ds, "test": val_ds})

# Option 1: upload to HF Hub (private)
HF_DATASET_ID = "akshan-main/restoredit-qwen-image-edit"

ddict.push_to_hub(HF_DATASET_ID, private=True)
print("Uploaded dataset to:", HF_DATASET_ID)

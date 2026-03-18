"""
Convert T-LESS BOP PBR annotations to COCO-format JSON for RT-DETR training.

BOP format stores annotations per scene in JSON files.
This script flattens everything into two COCO JSON files:
    data/tless_coco_train.json
    data/tless_coco_val.json

Run on the pfcalcul FRONTAL machine (CPU-only, fast ~2 min):
    python scripts/prepare_dataset.py

IMPORTANT: Edit the LOGIN variable below before running.
"""
import json
import os
import random
from pathlib import Path
from tqdm import tqdm

# ── EDIT THIS ──────────────────────────────────────────────────────────────────
LOGIN = "msabbah"   # your pfcalcul login name, e.g. "jdupont"
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_PBR_DIR = Path("/datasets/tless/train_pbr")
OUTPUT_DIR    = Path(f"/home/{LOGIN}/tless_detector/data")

NUM_OBJECTS = 30     # T-LESS has objects with obj_id 1..30
MIN_VISIB   = 0.10   # ignore instances with less than 10% visibility (BOP standard)
MIN_BBOX_PX = 5      # ignore bounding boxes smaller than 5 px on any side
VAL_SPLIT   = 0.05   # 5% of images go to validation
RANDOM_SEED = 42

# T-LESS PBR images are always 720×540
IMAGE_W, IMAGE_H = 720, 540


def build_categories():
    """COCO categories: id is 0-indexed (0 to 29) so RT-DETR can use it directly as a class index."""
    return [
        {"id": i - 1, "name": f"obj_{i:02d}", "supercategory": "tless"}
        for i in range(1, NUM_OBJECTS + 1)
    ]


def process_scene(scene_dir, global_image_id, global_ann_id):
    """
    Process one BOP scene folder and return COCO-format dicts.

    BOP scene_gt.json  → object poses keyed by image index string "0", "1", ...
    BOP scene_gt_info.json → bounding boxes and visibility, parallel to scene_gt
    """
    with open(scene_dir / "scene_gt.json") as f:
        scene_gt = json.load(f)
    with open(scene_dir / "scene_gt_info.json") as f:
        scene_gt_info = json.load(f)

    images      = []
    annotations = []

    for img_idx_str in sorted(scene_gt.keys(), key=lambda x: int(x)):
        rgb_path = scene_dir / "rgb" / f"{int(img_idx_str):06d}.jpg"
        if not rgb_path.exists():
            continue

        # file_name is relative to /datasets/tless — the img_folder in the config
        images.append({
            "id":        global_image_id,
            "file_name": str(rgb_path.relative_to(Path("/datasets/tless"))),
            "width":     IMAGE_W,
            "height":    IMAGE_H,
        })

        gt_list   = scene_gt[img_idx_str]
        info_list = scene_gt_info[img_idx_str]

        for gt, info in zip(gt_list, info_list):
            # Filter by visibility
            if info["visib_fract"] < MIN_VISIB:
                continue

            x, y, w, h = info["bbox_visib"]  # [x_min, y_min, width, height]

            # Clip to image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, IMAGE_W - x)
            h = min(h, IMAGE_H - y)

            if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
                continue

            annotations.append({
                "id":           global_ann_id,
                "image_id":     global_image_id,
                "category_id":  gt["obj_id"] - 1,  # 0-indexed (0–29) for RT-DETR class indices
                "bbox":         [x, y, w, h],   # COCO: [x_min, y_min, w, h]
                "area":         w * h,
                "iscrowd":      0,
                "segmentation": [],             # not used by RT-DETR
            })
            global_ann_id += 1

        global_image_id += 1

    return images, annotations, global_image_id, global_ann_id


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(TRAIN_PBR_DIR.iterdir())
    print(f"Found {len(scene_dirs)} scenes in {TRAIN_PBR_DIR}")

    all_images      = []
    all_annotations = []
    img_id = 1
    ann_id = 1

    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        imgs, anns, img_id, ann_id = process_scene(scene_dir, img_id, ann_id)
        all_images      += imgs
        all_annotations += anns

    print(f"\nTotal images:      {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")

    # Train / val split (reproducible with fixed seed)
    random.seed(RANDOM_SEED)
    val_count = int(len(all_images) * VAL_SPLIT)
    val_ids   = set(img["id"] for img in random.sample(all_images, val_count))

    categories = build_categories()

    splits = [
        ("train",
         [i for i in all_images      if i["id"] not in val_ids],
         [a for a in all_annotations if a["image_id"] not in val_ids]),
        ("val",
         [i for i in all_images      if i["id"] in val_ids],
         [a for a in all_annotations if a["image_id"] in val_ids]),
    ]

    for split_name, imgs, anns in splits:
        out_path = OUTPUT_DIR / f"tless_coco_{split_name}.json"
        coco_dict = {
            "info":        {"description": "T-LESS BOP PBR converted for RT-DETR"},
            "licenses":    [],
            "categories":  categories,
            "images":      imgs,
            "annotations": anns,
        }
        with open(out_path, "w") as f:
            json.dump(coco_dict, f)
        print(f"Saved {split_name:5s}: {len(imgs):6d} images, "
              f"{len(anns):7d} annotations → {out_path}")


if __name__ == "__main__":
    main()

"""
Visual sanity check: draws COCO bounding boxes on a few random training images.

Run interactively in JupyterLab on pfcalcul BEFORE launching training.
Open a terminal in JupyterLab and run:
    python scripts/verify_dataset.py

Output images are saved to ~/tless_detector/verify_output/
Open them with the JupyterLab file browser (just click to preview).

IMPORTANT: Edit the LOGIN variable below before running.
"""
import json
import random
from pathlib import Path

import cv2

# ── EDIT THIS ──────────────────────────────────────────────────────────────────
LOGIN = "msabbah"
# ──────────────────────────────────────────────────────────────────────────────

COCO_JSON    = Path(f"/home/{LOGIN}/tless_detector/data/tless_coco_train.json")
DATASET_ROOT = Path("/datasets/tless")
OUT_DIR      = Path(f"/home/{LOGIN}/tless_detector/verify_output")
N_SAMPLES    = 8   # number of images to check

OUT_DIR.mkdir(exist_ok=True)

print(f"Loading annotations from {COCO_JSON} ...")
with open(COCO_JSON) as f:
    coco = json.load(f)

# Build lookup: image_id → list of annotations
ann_by_img = {}
for ann in coco["annotations"]:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

print(f"Loaded {len(coco['images'])} images, {len(coco['annotations'])} annotations")
print(f"Sampling {N_SAMPLES} images ...\n")

for img_entry in random.sample(coco["images"], N_SAMPLES):
    img_path = DATASET_ROOT / img_entry["file_name"]
    frame    = cv2.imread(str(img_path))

    if frame is None:
        print(f"  [ERROR] Cannot open: {img_path}")
        continue

    anns = ann_by_img.get(img_entry["id"], [])

    for ann in anns:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        obj_id     = ann["category_id"]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label
        label = f"obj_{obj_id:02d}"
        cv2.putText(frame, label, (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    cv2.LINE_AA)

    # Save output
    out_path = OUT_DIR / f"verify_{img_entry['id']:06d}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"  {img_path.name}  →  {len(anns)} boxes  →  {out_path.name}")

print(f"\nDone. Check images in: {OUT_DIR}")
print("In JupyterLab: use the file browser on the left to navigate and click to preview.")

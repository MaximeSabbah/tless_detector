"""
Download the T-LESS BOP dataset to /datasets/tless/ on pfcalcul.

Run this ONCE on the pfcalcul FRONTAL machine (not inside a job):
    python data/download_tless.py

/datasets/ is the platform's dedicated dataset directory.
It is automatically synchronized to the compute node's fast local
storage by the datasync tool when a job starts.

Do NOT run this inside a SLURM job — the frontal has internet access,
compute nodes typically do not.
"""
import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# pfcalcul: /datasets/ is the correct location for training data
DATASET_DIR = Path("/datasets/tless")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# pfcalcul policy: every dataset folder must have a README
readme = DATASET_DIR / "README.md"
if not readme.exists():
    readme.write_text(
        "# T-LESS BOP Dataset\n\n"
        "Downloaded for RT-DETR training (2D detection front-end for FoundationPose).\n\n"
        "Source: https://bop.felk.cvut.cz/datasets/ (HuggingFace: bop-benchmark/tless)\n\n"
        "Contact: <YOUR_NAME>@laas.fr\n"
    )
    print(f"Created README at {readme}")

# Files to download.
# tless_train_pbr.zip is ~15 GB — be patient.
FILES = [
    "tless_base.zip",       # metadata, camera parameters (~1 MB)
    "tless_train_pbr.zip",  # PBR synthetic training images (~15 GB)
    # "tless_models.zip",   # 3D CAD models — uncomment if you want to
    #                         convert meshes directly on pfcalcul
]

for fname in FILES:
    zip_path = DATASET_DIR / fname
    marker   = DATASET_DIR / (fname.replace(".zip", "") + ".extracted")

    if marker.exists():
        print(f"[SKIP] Already extracted: {fname}")
        continue

    if not zip_path.exists():
        print(f"[DOWNLOAD] {fname} ...")
        hf_hub_download(
            repo_id="bop-benchmark/tless",
            filename=fname,
            repo_type="dataset",
            local_dir=str(DATASET_DIR),
        )
        print(f"[DOWNLOAD] Done: {fname}")
    else:
        print(f"[EXTRACT]  {fname} (zip already present) ...")

    print(f"[EXTRACT]  {fname} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATASET_DIR)
    marker.touch()
    os.remove(zip_path)
    print(f"[DONE]     {fname}")

print()
print(f"Dataset ready at: {DATASET_DIR}")
print()
print("Expected structure:")
print("  /datasets/tless/camera_primesense.json")
print("  /datasets/tless/train_pbr/000000/ ... 000049/")
print("    Each scene folder contains:")
print("      rgb/          RGB images")
print("      scene_gt.json          object poses per image")
print("      scene_gt_info.json     bounding boxes + visibility")

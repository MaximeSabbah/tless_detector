"""
Download the T-LESS BOP dataset.

--- Full training dataset (default) ---
Run ONCE on the pfcalcul FRONTAL machine (not inside a job):
    python data/download_tless.py

/datasets/ is the platform's dedicated dataset directory.
It is automatically synchronized to the compute node's fast local
storage by the datasync tool when a job starts.

Do NOT run this inside a SLURM job — the frontal has internet access,
compute nodes typically do not.

--- Meshes only (for FoundationPose) ---
Run on any machine with internet access and trimesh installed:
    python data/download_tless.py --meshes-only [--out-dir /your/path]

This downloads only tless_models.zip (~100 MB) and extracts the .ply
files. Then run scripts/convert_meshes.py to produce .obj files ready
for FoundationPose.
"""
import argparse
import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download T-LESS BOP dataset (full or meshes only)."
    )
    parser.add_argument(
        "--meshes-only",
        action="store_true",
        help=(
            "Download only tless_models.zip (3D CAD models, ~100 MB) "
            "instead of the full training dataset (~15 GB). "
            "Use this to prepare meshes for FoundationPose."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for downloaded files. "
            "Defaults to /datasets/tless for full download, "
            "or ./tless_meshes for --meshes-only."
        ),
    )
    return parser.parse_args()


def download_and_extract(fname, dataset_dir):
    zip_path = dataset_dir / fname
    marker   = dataset_dir / (fname.replace(".zip", "") + ".extracted")

    if marker.exists():
        print(f"[SKIP] Already extracted: {fname}")
        return

    if not zip_path.exists():
        print(f"[DOWNLOAD] {fname} ...")
        hf_hub_download(
            repo_id="bop-benchmark/tless",
            filename=fname,
            repo_type="dataset",
            local_dir=str(dataset_dir),
        )
        print(f"[DOWNLOAD] Done: {fname}")
    else:
        print(f"[EXTRACT]  {fname} (zip already present) ...")

    print(f"[EXTRACT]  {fname} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    marker.touch()
    os.remove(zip_path)
    print(f"[DONE]     {fname}")


def main():
    args = parse_args()

    if args.meshes_only:
        dataset_dir = args.out_dir or Path("./tless_meshes")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print("Meshes-only mode: downloading tless_models.zip (~100 MB)")
        print(f"Output directory: {dataset_dir.resolve()}")
        print()

        download_and_extract("tless_models.zip", dataset_dir)

        print()
        print(f"Meshes extracted to: {dataset_dir.resolve()}")
        print()
        print("Next step — convert to .obj for FoundationPose:")
        print(f"  python scripts/convert_meshes.py --models-dir {dataset_dir}/models_cad")
        print()
        print("Tip: models_cad/ is recommended for FoundationPose (original CAD geometry).")
        print("     models_eval/ and models_reconst/ are also available.")

    else:
        # Full training dataset download (pfcalcul)
        dataset_dir = args.out_dir or Path("/datasets/tless")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # pfcalcul policy: every dataset folder must have a README
        readme = dataset_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# T-LESS BOP Dataset\n\n"
                "Downloaded for RT-DETR training (2D detection front-end for FoundationPose).\n\n"
                "Source: https://bop.felk.cvut.cz/datasets/ (HuggingFace: bop-benchmark/tless)\n\n"
                "Contact: <msabbah>@laas.fr\n"
            )
            print(f"Created README at {readme}")

        # tless_train_pbr.zip is ~15 GB — be patient.
        FILES = [
            "tless_base.zip",       # metadata, camera parameters (~1 MB)
            "tless_train_pbr.zip",  # PBR synthetic training images (~15 GB)
        ]

        for fname in FILES:
            download_and_extract(fname, dataset_dir)

        print()
        print(f"Dataset ready at: {dataset_dir}")
        print()
        print("Expected structure:")
        print(f"  {dataset_dir}/camera_primesense.json")
        print(f"  {dataset_dir}/train_pbr/000000/ ... 000049/")
        print("    Each scene folder contains:")
        print("      rgb/                RGB images")
        print("      scene_gt.json       object poses per image")
        print("      scene_gt_info.json  bounding boxes + visibility")


if __name__ == "__main__":
    main()

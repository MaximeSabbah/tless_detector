"""
Convert T-LESS CAD models from .ply to .obj, centering the mesh at its centroid.

FoundationPose requires the mesh origin to be at the center of the object.
The T-LESS models ship as .ply files; this script converts them to .obj.

--- Quick start ---

1. Download the meshes (if not already done):
       python data/download_tless.py --meshes-only --out-dir ./tless_meshes

2. Convert for FoundationPose:
       python scripts/convert_meshes.py

3. Copy .obj files to the Isaac ROS machine:
       scp data/tless_obj_meshes/*.obj user@robot:${ISAAC_ROS_WS}/isaac_ros_assets/tless_meshes/

--- T-LESS model variants ---

tless_models.zip contains three subdirectories:
  models_cad/     Original CAD models — best for FoundationPose (default)
  models_eval/    Simplified models used for BOP evaluation metrics
  models_reconst/ Reconstructed from real scans

--- Options ---

    --models-dir PATH   Directory containing T-LESS .ply files.
                        Default: ./tless_meshes/models_cad
    --out-dir PATH      Output directory for .obj files.
                        Default: data/tless_obj_meshes

Requirements:
    pip install trimesh
"""
import argparse
from pathlib import Path

import trimesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert T-LESS .ply meshes to .obj for FoundationPose."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./tless_meshes/models_cad"),
        help=(
            "Directory containing T-LESS obj_XXXXXX.ply files. "
            "Use models_cad/ (default), models_eval/, or models_reconst/ "
            "from the extracted tless_models.zip. "
            "Default: ./tless_meshes/models_cad"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tless_obj_meshes"),
        help=(
            "Output directory for converted .obj files. "
            "Default: data/tless_obj_meshes"
        ),
    )
    return parser.parse_args()


def convert_all(models_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(models_dir.glob("obj_*.ply"))

    if not ply_files:
        print(f"ERROR: No .ply files found in {models_dir}")
        print("Did you download tless_models.zip ?")
        print("  python data/download_tless.py --meshes-only --out-dir ./tless_meshes")
        print()
        print("Available model variants (pass with --models-dir):")
        print("  ./tless_meshes/models_cad      ← recommended for FoundationPose")
        print("  ./tless_meshes/models_eval")
        print("  ./tless_meshes/models_reconst")
        return

    print(f"Found {len(ply_files)} models in {models_dir}")
    print(f"Output: {out_dir.resolve()}")
    print()

    for ply_path in ply_files:
        mesh     = trimesh.load(str(ply_path))
        centroid = mesh.centroid.copy()

        # Scale from millimetres (T-LESS native unit) to metres (FoundationPose)
        mesh.apply_scale(0.001)

        # Translate so the bounding-box centre is at origin (0, 0, 0).
        # FoundationPose needs the mesh origin at the visual centre of the object.
        # Bounding-box centre is a better approximation than the centroid (CoM)
        # for asymmetric objects like many T-LESS parts.
        bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-bbox_center)

        out_path = out_dir / ply_path.with_suffix(".obj").name
        mesh.export(str(out_path))

        print(f"  {ply_path.name}"
              f"  centroid_offset={centroid.round(3)}"
              f"  →  {out_path.name}")

    print(f"\nAll {len(ply_files)} meshes converted to: {out_dir.resolve()}")
    print()
    print("Copy these files to your Isaac ROS machine, for example:")
    print(f"  scp {out_dir}/*.obj user@robot:${{ISAAC_ROS_WS}}/isaac_ros_assets/tless_meshes/")


if __name__ == "__main__":
    args = parse_args()
    convert_all(args.models_dir, args.out_dir)

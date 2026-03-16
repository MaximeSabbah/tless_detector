"""
Convert T-LESS CAD models from .ply to .obj, centering the mesh at its centroid.

FoundationPose requires the mesh origin to be at the center of the object.
The T-LESS models ship as .ply files; this script converts them to .obj.

Run once (locally or on pfcalcul frontal):
    python scripts/convert_meshes.py

Output goes to:  data/tless_obj_meshes/obj_000001.obj ... obj_000030.obj

You then need to copy these .obj files to your Isaac ROS machine.

Requirements:
    pip install trimesh

IMPORTANT: Edit the PLY_DIR and OUT_DIR variables below if needed.
           If you are running on pfcalcul, download tless_models.zip first
           (uncomment that line in data/download_tless.py).
"""
import trimesh
from pathlib import Path

# Input: T-LESS .ply models (download tless_models.zip first)
PLY_DIR = Path("/datasets/tless/models")

# Output: .obj files ready for FoundationPose
OUT_DIR = Path("data/tless_obj_meshes")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_all():
    ply_files = sorted(PLY_DIR.glob("obj_*.ply"))

    if not ply_files:
        print(f"ERROR: No .ply files found in {PLY_DIR}")
        print("Did you download tless_models.zip ?")
        print("Uncomment that line in data/download_tless.py and re-run it.")
        return

    print(f"Found {len(ply_files)} models to convert.\n")

    for ply_path in ply_files:
        mesh     = trimesh.load(str(ply_path))
        centroid = mesh.centroid.copy()

        # Translate mesh so that centroid is at origin (0, 0, 0)
        mesh.apply_translation(-centroid)

        out_path = OUT_DIR / ply_path.with_suffix(".obj").name
        mesh.export(str(out_path))

        print(f"  {ply_path.name}"
              f"  centroid_offset={centroid.round(3)}"
              f"  →  {out_path.name}")

    print(f"\nAll {len(ply_files)} meshes converted to: {OUT_DIR.resolve()}")
    print()
    print("Copy these files to your Isaac ROS machine, for example:")
    print("  scp data/tless_obj_meshes/*.obj user@robot:${ISAAC_ROS_WS}/isaac_ros_assets/tless_meshes/")


if __name__ == "__main__":
    convert_all()

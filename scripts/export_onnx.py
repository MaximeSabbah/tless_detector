"""
Export a trained RT-DETR v2 checkpoint to ONNX format.

This script:
1. Calls the official RT-DETR v2 export tool
2. Simplifies the ONNX graph with onnxsim
3. Renames output tensors to the names expected by isaac_ros_rtdetr:
       input:  "images"
       outputs: "labels", "boxes", "scores"
4. Runs a dummy forward pass to verify shapes

Run on the pfcalcul frontal after training:
    cd ~/tless_detector
    source .venv/bin/activate
    python scripts/export_onnx.py \
        --checkpoint output/rtdetr_r50vd_tless/best.pth \
        --output     output/tless_rtdetr.onnx
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim

REPO = Path(__file__).resolve().parent.parent

# These are the exact names isaac_ros_rtdetr expects at the TRT binding level.
EXPECTED_INPUT   = "images"
EXPECTED_OUTPUTS = ["labels", "boxes", "scores"]

INPUT_SIZE = 640  # must match eval_spatial_size in the config


def run_official_export(checkpoint: Path, config: Path, out_raw: Path):
    """Use the official RT-DETR v2 export script."""
    export_script = REPO / "third_party/RT-DETR/rtdetrv2_pytorch/tools/export_onnx.py"

    if not export_script.exists():
        raise FileNotFoundError(
            f"RT-DETR v2 export script not found at {export_script}.\n"
            "Did you run: git submodule update --init --recursive ?"
        )

    cmd = [
        sys.executable, str(export_script),
        "-c", str(config),
        "-r", str(checkpoint),
        "-o", str(out_raw),
        "-s", str(INPUT_SIZE),
        "--check",
    ]
    print(f"\n[EXPORT] Running RT-DETR v2 export ...")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "Export failed. Check the output above for details.\n"
            "Common causes:\n"
            "  - Wrong checkpoint path\n"
            "  - num_classes mismatch between config and checkpoint\n"
            "  - PyTorch version incompatibility"
        )
    print("[EXPORT] Done.")


def simplify(src: Path, dst: Path):
    """Fold constants and simplify the ONNX graph (makes TRT conversion faster)."""
    print(f"\n[SIMPLIFY] {src.name} → {dst.name} ...")
    model = onnx.load(str(src))
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        print("[SIMPLIFY] WARNING: onnxsim could not fully verify the model.")
        print("           Proceeding anyway — the model may still work.")
    onnx.save(simplified, str(dst))
    print("[SIMPLIFY] Done.")


def inspect(path: Path):
    """Print input/output tensor names and shapes for debugging."""
    model = onnx.load(str(path))
    print(f"\n[INSPECT] {path.name}")
    print("  Inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    name='{inp.name}'  shape={shape}")
    print("  Outputs:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    name='{out.name}'  shape={shape}")


def rename_if_needed(path: Path):
    """
    Rename input/output tensors to what isaac_ros_rtdetr expects.
    The official export usually produces the correct names already,
    but this is a safety net in case they differ.
    """
    model = onnx.load(str(path))

    current_in  = [i.name for i in model.graph.input]
    current_out = [o.name for o in model.graph.output]

    rename = {}
    if current_in[0] != EXPECTED_INPUT:
        rename[current_in[0]] = EXPECTED_INPUT
    for actual, expected in zip(current_out, EXPECTED_OUTPUTS):
        if actual != expected:
            rename[actual] = expected

    if not rename:
        print("\n[RENAME] Tensor names already correct. No renaming needed.")
        return

    print(f"\n[RENAME] Renaming: {rename}")

    for node in model.graph.node:
        node.input[:]  = [rename.get(n, n) for n in node.input]
        node.output[:] = [rename.get(n, n) for n in node.output]
    for inp in model.graph.input:
        if inp.name in rename: inp.name = rename[inp.name]
    for out in model.graph.output:
        if out.name in rename: out.name = rename[out.name]

    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print("[RENAME] Done.")


def verify_inference(path: Path):
    """
    Run a dummy forward pass with ONNXRuntime.
    Confirms the model loads correctly and output shapes match expectations.
    """
    print(f"\n[VERIFY] Running dummy inference with ONNXRuntime ...")
    sess = ort.InferenceSession(
        str(path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    dummy   = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    size    = np.array([[INPUT_SIZE, INPUT_SIZE]], dtype=np.int64)
    inputs  = {EXPECTED_INPUT: dummy, "orig_target_sizes": size}
    outputs = sess.run(None, inputs)
    names   = [o.name for o in sess.get_outputs()]

    print("  Output tensors:")
    for name, arr in zip(names, outputs):
        print(f"    '{name}'  shape={arr.shape}  dtype={arr.dtype}")

    assert outputs[0].shape == (1, 300),    \
        f"'labels' shape should be (1, 300), got {outputs[0].shape}"
    assert outputs[1].shape == (1, 300, 4), \
        f"'boxes' shape should be (1, 300, 4), got {outputs[1].shape}"
    assert outputs[2].shape == (1, 300),    \
        f"'scores' shape should be (1, 300), got {outputs[2].shape}"

    print("[VERIFY] ✓ All shape checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained RT-DETR v2 to ONNX for isaac_ros_rtdetr"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--config",
                        default=str(REPO / "configs/rtdetr_r50vd_tless.yml"),
                        help="Path to training config YAML")
    parser.add_argument("--output", required=True,
                        help="Path for final .onnx output file")
    args = parser.parse_args()

    ckpt   = Path(args.checkpoint)
    config = Path(args.config)
    out    = Path(args.output)
    raw    = out.with_suffix(".raw.onnx")

    out.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    run_official_export(ckpt, config, raw)
    simplify(raw, out)
    raw.unlink(missing_ok=True)
    inspect(out)
    rename_if_needed(out)
    inspect(out)
    verify_inference(out)

    print(f"\n{'='*50}")
    print(f"Final ONNX saved at: {out}")
    print(f"{'='*50}")
    print("\nNext step: copy to your Isaac ROS machine and run trtexec.")
    print("See cluster/README.md for the exact commands.")


if __name__ == "__main__":
    main()

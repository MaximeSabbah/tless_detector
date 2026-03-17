"""
Run inference with a fine-tuned RT-DETR v2 checkpoint on unseen images.

Draws bounding boxes with class names and confidence scores, saves annotated
images to the output directory.

Usage (from repo root):
    source .venv/bin/activate
    python scripts/predict.py \
        --checkpoint output/rtdetr_r50vd_tless/best.pth \
        --images     /path/to/test/images/ \
        --output     output/predictions/ \
        --threshold  0.5
"""
import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party/RT-DETR/rtdetrv2_pytorch"))

from src.core import YAMLConfig  # noqa: E402

# obj_01 … obj_30 — T-LESS object names match the category IDs in the COCO JSON
TLESS_NAMES = {i: f"obj_{i:02d}" for i in range(1, 31)}

# Distinct colours for up to 30 classes
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffffff", "#000000", "#e6beff", "#fabebe", "#008080",
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
]


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    cfg = YAMLConfig(str(config_path))
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Info: {len(missing)} keys not loaded (expected — class heads differ from pretrain)")

    return model, postprocessor


@torch.no_grad()
def predict(model, postprocessor, image_path: Path, device: str,
            input_size: int = 640, threshold: float = 0.5):
    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size

    # Resize to model input size and convert to tensor
    resized = orig.resize((input_size, input_size), Image.BILINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)           # [1, 3, H, W]

    # orig_target_sizes: [W, H] — matches the format used in coco_dataset.py
    orig_size = torch.tensor([[orig_w, orig_h]], device=device)

    outputs = model(tensor)
    results = postprocessor(outputs, orig_size)   # list of dicts: labels, boxes, scores

    labels = results[0]["labels"]   # [300]
    boxes  = results[0]["boxes"]    # [300, 4] absolute [x1, y1, x2, y2]
    scores = results[0]["scores"]   # [300]

    mask = scores >= threshold
    return labels[mask], boxes[mask], scores[mask], orig


def draw_detections(image: Image.Image, labels, boxes, scores) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        color = PALETTE[int(label) % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        name = TLESS_NAMES.get(int(label) + 1, f"cls_{int(label)}")
        draw.text((x1 + 2, y1 + 2), f"{name} {score:.2f}", fill=color)
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Run RT-DETR v2 inference on unseen T-LESS images"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--config",
                        default=str(REPO / "configs/rtdetr_r50vd_tless.yml"),
                        help="Path to training config YAML")
    parser.add_argument("--images", required=True,
                        help="Directory containing .jpg/.png test images")
    parser.add_argument("--output", default="output/predictions",
                        help="Directory to save annotated images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model on {args.device} ...")
    model, postprocessor = load_model(
        Path(args.config), Path(args.checkpoint), args.device
    )
    print("Model ready.\n")

    image_paths = sorted(Path(args.images).glob("*.jpg")) + \
                  sorted(Path(args.images).glob("*.png"))

    if not image_paths:
        print(f"No .jpg or .png images found in {args.images}")
        return

    for p in image_paths:
        labels, boxes, scores, orig = predict(
            model, postprocessor, p, args.device, threshold=args.threshold
        )
        annotated = draw_detections(orig, labels, boxes, scores)
        out_path = out_dir / p.name
        annotated.save(str(out_path))
        print(f"  {p.name}: {len(labels)} detection(s) → {out_path}")

    print(f"\nDone. Open {out_dir}/ in JupyterLab to inspect results.")


if __name__ == "__main__":
    main()

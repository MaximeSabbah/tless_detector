"""
Run inference with the TensorRT RT-DETR engine directly on an image.

Replicates the exact preprocessing the ROS pipeline uses:
  1. Center-crop to 4:3
  2. Resize to 640×480
  3. Pad to 640×640 (bottom-right, black)
  4. Normalize to [0, 1]
  orig_target_sizes is set to [640, 480] to match the ROS pipeline.

Usage (system Python3 — NOT the venv, which has no tensorrt):
    python3 scripts/predict_trt.py --image data/ima_b21d9f4.jpg

Options:
    --engine PATH     TRT .plan file  [default: isaac_ros_assets/models/tless/tless_rtdetr.plan]
    --image PATH      Input image
    --output PATH     Annotated output image  [default: /tmp/trt_pred.jpg]
    --threshold FLOAT Confidence threshold  [default: 0.1]
    --no-crop         Skip center-crop (use if image is already 4:3 or 1:1)
"""
import argparse
import ctypes
import os
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

# ── CUDA helpers via libcudart ───────────────────────────────────────────────
_cudart = ctypes.CDLL('libcudart.so')
_cudart.cudaMalloc.restype       = ctypes.c_int
_cudart.cudaMalloc.argtypes      = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaFree.restype         = ctypes.c_int
_cudart.cudaFree.argtypes        = [ctypes.c_void_p]
_cudart.cudaMemcpy.restype       = ctypes.c_int
_cudart.cudaMemcpy.argtypes      = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.c_int]
_cudart.cudaStreamCreate.restype  = ctypes.c_int
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamSynchronize.restype  = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
    return ptr


def cuda_free(ptr):
    _cudart.cudaFree(ptr)


def host_to_device(host_arr: np.ndarray, gpu_ptr: ctypes.c_void_p):
    _cudart.cudaMemcpy(gpu_ptr, host_arr.ctypes.data,
                       host_arr.nbytes, cudaMemcpyHostToDevice)


def device_to_host(host_arr: np.ndarray, gpu_ptr: ctypes.c_void_p):
    _cudart.cudaMemcpy(host_arr.ctypes.data, gpu_ptr,
                       host_arr.nbytes, cudaMemcpyDeviceToHost)


# ── Preprocessing ────────────────────────────────────────────────────────────

def center_crop_to_43(img: np.ndarray) -> np.ndarray:
    """Center-crop BGR image to 4:3 aspect ratio."""
    h, w = img.shape[:2]
    target_ar = 4 / 3
    if w / h > target_ar:
        new_w = int(h * target_ar)
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ar)
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]


def preprocess(image_path: Path, crop: bool = True):
    """
    Return (input_chw_float32, annotatable_640x480_bgr).

    Matches the ROS pipeline:
      crop → resize 640×480 → pad to 640×640 → normalize [0,1] → CHW
    orig_target_sizes is always [640, 480].
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f'Cannot read: {image_path}')

    if crop:
        bgr = center_crop_to_43(bgr)

    # Step 1: resize to 640×480 (matches create_tless_bag.py output)
    bgr_480 = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)

    # Step 2: pad bottom-right to 640×640 (matches pad_node in ROS)
    bgr_640 = np.zeros((640, 640, 3), dtype=np.uint8)
    bgr_640[:480, :640] = bgr_480

    # Step 3: BGR → RGB, normalize to [0, 1], CHW, float32
    rgb = cv2.cvtColor(bgr_640, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])  # 1×3×640×640

    return chw, bgr_480   # return the 640×480 image for annotation


# ── TRT inference ────────────────────────────────────────────────────────────

def load_engine(plan_path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(plan_path, 'rb') as f:
        return trt.Runtime(logger).deserialize_cuda_engine(f.read())


def run_inference(engine, image_chw: np.ndarray):
    context = engine.create_execution_context()

    # orig_target_sizes: [W, H] of the image after the 640×480 resize step
    orig_sizes = np.array([[640, 480]], dtype=np.int64)

    # Allocate GPU buffers
    gpu = {
        'images':            cuda_malloc(image_chw.nbytes),
        'orig_target_sizes': cuda_malloc(orig_sizes.nbytes),
        'labels':            cuda_malloc(1 * 300 * np.dtype(np.int64).itemsize),
        'boxes':             cuda_malloc(1 * 300 * 4 * np.dtype(np.float32).itemsize),
        'scores':            cuda_malloc(1 * 300 * np.dtype(np.float32).itemsize),
    }

    host_out = {
        'labels': np.empty((1, 300),    dtype=np.int64),
        'boxes':  np.empty((1, 300, 4), dtype=np.float32),
        'scores': np.empty((1, 300),    dtype=np.float32),
    }

    # Copy inputs to GPU
    host_to_device(image_chw, gpu['images'])
    host_to_device(orig_sizes, gpu['orig_target_sizes'])

    # Bind tensor addresses (TRT expects a plain int, not ctypes.c_void_p)
    for name, ptr in gpu.items():
        context.set_tensor_address(name, ptr.value)

    # Run
    stream = ctypes.c_void_p()
    _cudart.cudaStreamCreate(ctypes.byref(stream))
    context.execute_async_v3(stream.value)
    _cudart.cudaStreamSynchronize(stream.value)

    # Copy outputs back
    for name in host_out:
        device_to_host(host_out[name], gpu[name])

    for ptr in gpu.values():
        cuda_free(ptr)

    return host_out['labels'][0], host_out['boxes'][0], host_out['scores'][0]


# ── Visualisation ────────────────────────────────────────────────────────────

TLESS_NAMES = {i: f'obj_{i:02d}' for i in range(1, 31)}

PALETTE = [
    (230, 25,  75),  (60,  180, 75),  (255, 225, 25),  (67,  99,  216),
    (245, 130, 49),  (145, 30,  180), (66,  212, 244), (240, 50,  230),
    (191, 239, 69),  (250, 190, 212), (70,  153, 144), (220, 190, 255),
    (154, 99,  36),  (255, 250, 200), (128, 0,   0),   (170, 255, 195),
    (128, 128, 0),   (255, 216, 177), (0,   0,   117), (169, 169, 169),
]


def draw_detections(bgr: np.ndarray, labels, boxes, scores, threshold: float):
    out = bgr.copy()
    h, w = out.shape[:2]

    # boxes are scaled to orig_target_sizes=[640,480] → already in pixel coords
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = PALETTE[int(label) % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        name = TLESS_NAMES.get(int(label) + 1, f'cls_{int(label)}')
        text = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros_ws')

def main():
    parser = argparse.ArgumentParser(
        description='Test the TensorRT RT-DETR engine directly on an image.'
    )
    parser.add_argument('--engine', type=Path,
                        default=Path(f'{ISAAC_ROS_WS}/isaac_ros_assets/models/tless/tless_rtdetr.plan'),
                        help='TensorRT .plan file')
    parser.add_argument('--image', type=Path, required=True,
                        help='Input image')
    parser.add_argument('--output', type=Path, default=Path('/tmp/trt_pred.jpg'),
                        help='Annotated output image  [default: /tmp/trt_pred.jpg]')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Confidence threshold  [default: 0.85]')
    parser.add_argument('--no-crop', action='store_true',
                        help='Skip center-crop (image is already 4:3 or 1:1)')
    args = parser.parse_args()

    print(f'Loading engine: {args.engine}')
    engine = load_engine(args.engine)
    print('Engine loaded.\n')

    print(f'Preprocessing: {args.image}')
    image_chw, bgr_480 = preprocess(args.image, crop=not args.no_crop)
    print(f'  Input tensor: {image_chw.shape}  orig_target_sizes: [640, 480]\n')

    print('Running TRT inference...')
    labels, boxes, scores = run_inference(engine, image_chw)

    # Print all detections above a very low threshold to show score distribution
    above = [(float(scores[i]), int(labels[i]) + 1, boxes[i].tolist())
             for i in range(len(scores)) if scores[i] >= 0.01]
    above.sort(key=lambda x: x[0], reverse=True)

    print(f'Detections (all scores ≥ 0.01):')
    if not above:
        print('  None — model produced no output above 0.01.')
    for score, cls_id, box in above[:20]:
        marker = ' ✓' if score >= args.threshold else ''
        print(f'  obj_{cls_id:02d}  score={score:.3f}  box={[round(v) for v in box]}{marker}')

    passing = sum(1 for s, _, _ in above if s >= args.threshold)
    print(f'\n{passing} detection(s) above threshold {args.threshold}')

    out = draw_detections(bgr_480, labels, boxes, scores, args.threshold)
    cv2.imwrite(str(args.output), out)
    print(f'Annotated image saved to: {args.output}')


if __name__ == '__main__':
    main()

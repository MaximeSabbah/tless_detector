"""
Compare PyTorch / ONNX / TensorRT outputs on the same image with identical preprocessing.

Run from the venv (has torch + onnxruntime; TRT is pulled from system Python):
    source .venv/bin/activate
    python scripts/compare_backends.py --image data/ima_b21d9f4.jpg

All three backends use the same preprocessing as predict.py:
    resize to 640×640 → normalize [0, 1] → orig_target_sizes = [orig_w, orig_h]
so score differences are purely due to the conversion, not preprocessing.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Make system-level TensorRT importable from inside the venv
sys.path.insert(0, '/usr/lib/python3.12/dist-packages')

REPO       = Path(__file__).resolve().parent.parent
ISAAC_WS   = os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros_ws')
ONNX_PATH  = Path(f'{ISAAC_WS}/tless_detector/output/rtdetr_r50vd_tless/tless_rtdetr.onnx')
TRT_PATH   = Path(f'{ISAAC_WS}/isaac_ros_assets/models/tless/tless_rtdetr.plan')
CFG_PATH   = REPO / 'configs/rtdetr_r50vd_tless.yml'
CKPT_PATH  = REPO / 'output/rtdetr_r50vd_tless/best.pth'
TLESS_NAMES = {i: f'obj_{i:02d}' for i in range(1, 31)}


# ── Shared preprocessing ─────────────────────────────────────────────────────
# Matches predict.py exactly: resize to 640×640, normalize [0,1], no crop.
# orig_target_sizes = [orig_w, orig_h] so boxes are in original image coordinates.

def preprocess(image_path: Path):
    bgr = cv2.imread(str(image_path))
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]  # 1×3×640×640
    orig_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)               # [W, H]
    return chw, orig_sizes, orig_w, orig_h


# ── PyTorch backend ──────────────────────────────────────────────────────────

def run_pth(chw, orig_sizes):
    import torch
    sys.path.insert(0, str(REPO / 'third_party/RT-DETR/rtdetrv2_pytorch'))
    from src.core import YAMLConfig

    cfg   = YAMLConfig(str(CFG_PATH))
    model = cfg.model.cuda().eval()
    post  = cfg.postprocessor.cuda().eval()

    ckpt  = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    state = ckpt.get('ema', {}).get('module') or ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)

    t_img  = torch.from_numpy(chw).cuda()
    t_size = torch.from_numpy(orig_sizes.astype(np.int64)).cuda()

    with torch.no_grad():
        out = post(model(t_img), t_size)

    return (out[0]['labels'].cpu().numpy(),
            out[0]['boxes'].cpu().numpy(),
            out[0]['scores'].cpu().numpy())


# ── ONNX backend (onnxruntime) ───────────────────────────────────────────────

def run_onnx(chw, orig_sizes):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=['CUDAExecutionProvider',
                                           'CPUExecutionProvider'])
    labels, boxes, scores = sess.run(
        ['labels', 'boxes', 'scores'],
        {'images': chw, 'orig_target_sizes': orig_sizes},
    )
    return labels[0], boxes[0], scores[0]


# ── TensorRT backend ─────────────────────────────────────────────────────────

def run_trt(chw, orig_sizes):
    import ctypes
    import tensorrt as trt

    cudart = ctypes.CDLL('libcudart.so')
    cudart.cudaMalloc.restype       = ctypes.c_int
    cudart.cudaMalloc.argtypes      = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.restype         = ctypes.c_int
    cudart.cudaFree.argtypes        = [ctypes.c_void_p]
    cudart.cudaMemcpy.restype       = ctypes.c_int
    cudart.cudaMemcpy.argtypes      = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int]
    cudart.cudaStreamCreate.restype  = ctypes.c_int
    cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaStreamSynchronize.restype  = ctypes.c_int
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    H2D, D2H = 1, 2

    with open(TRT_PATH, 'rb') as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    bufs, host = {}, {}
    specs = {
        'images':            (chw,        np.float32, (1, 3, 640, 640)),
        'orig_target_sizes': (orig_sizes,  np.int64,   (1, 2)),
        'labels':            (None,        np.int64,   (1, 300)),
        'boxes':             (None,        np.float32, (1, 300, 4)),
        'scores':            (None,        np.float32, (1, 300)),
    }
    for name, (data, dtype, shape) in specs.items():
        arr = np.ascontiguousarray(data if data is not None else np.empty(shape, dtype))
        ptr = ctypes.c_void_p()
        cudart.cudaMalloc(ctypes.byref(ptr), arr.nbytes)
        if data is not None:
            cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes, H2D)
        bufs[name] = ptr
        host[name] = np.empty(shape, dtype=dtype) if data is None else arr

    for name, ptr in bufs.items():
        ctx.set_tensor_address(name, ptr.value)

    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))
    ctx.execute_async_v3(stream.value)
    cudart.cudaStreamSynchronize(stream.value)

    for name in ('labels', 'boxes', 'scores'):
        cudart.cudaMemcpy(host[name].ctypes.data, bufs[name], host[name].nbytes, D2H)
    for ptr in bufs.values():
        cudart.cudaFree(ptr)

    return host['labels'][0], host['boxes'][0], host['scores'][0]


# ── Display ──────────────────────────────────────────────────────────────────

def top_detections(labels, boxes, scores, n=10):
    order = np.argsort(scores)[::-1][:n]
    return [(float(scores[i]), TLESS_NAMES.get(int(labels[i]) + 1, f'cls_{int(labels[i])}'),
             [round(float(v)) for v in boxes[i]]) for i in order if scores[i] > 0.05]


def print_results(name, detections):
    print(f'\n  {"─"*55}')
    print(f'  {name}')
    print(f'  {"─"*55}')
    if not detections:
        print('  (no detections above 0.05)')
        return
    for score, cls, box in detections:
        print(f'  {cls}  {score:.3f}  {box}')


def save_annotated(image_path, labels, boxes, scores, out_path, threshold):
    bgr = cv2.imread(str(image_path))
    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{TLESS_NAMES.get(int(label)+1,"?")} {score:.2f}'
        cv2.putText(bgr, text, (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out_path), bgr)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare pth / onnx / trt on the same image.')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/compare'))
    parser.add_argument('--backends', default='pth,onnx,trt',
                        help='Comma-separated subset: pth,onnx,trt')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(',')]

    print(f'Image: {args.image}')
    chw, orig_sizes, orig_w, orig_h = preprocess(args.image)
    print(f'Preprocessed: 1×3×640×640  orig_target_sizes=[{orig_w}, {orig_h}]')

    runners = {'pth': run_pth, 'onnx': run_onnx, 'trt': run_trt}
    results = {}

    for name in backends:
        print(f'\nRunning {name.upper()} ...', end=' ', flush=True)
        try:
            labels, boxes, scores = runners[name](chw, orig_sizes)
            results[name] = (labels, boxes, scores)
            print('done')
            print_results(name.upper(), top_detections(labels, boxes, scores))
            out = args.output_dir / f'{name}.jpg'
            save_annotated(args.image, labels, boxes, scores, out, args.threshold)
            print(f'  → {out}')
        except Exception as e:
            print(f'FAILED: {e}')

    # Score comparison for obj_23 specifically
    if len(results) > 1:
        print(f'\n  {"─"*55}')
        print('  obj_23 score comparison')
        print(f'  {"─"*55}')
        for name, (labels, boxes, scores) in results.items():
            mask = (labels + 1) == 23
            best = float(scores[mask].max()) if mask.any() else 0.0
            print(f'  {name.upper():<6}  {best:.4f}')


if __name__ == '__main__':
    main()

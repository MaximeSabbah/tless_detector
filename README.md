# T-LESS RT-DETR Detector for FoundationPose

Fine-tuned RT-DETR object detector for [T-LESS](https://cmp.felk.cvut.cz/t-less/) industrial objects,
intended to be used as the 2D detection front-end for
[NVIDIA Isaac ROS FoundationPose](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html).

## What this repo does

1. Downloads the T-LESS BOP dataset (PBR synthetic images)
2. Converts BOP annotations to COCO format for RT-DETR training
3. Fine-tunes RT-DETR v2 (r50vd backbone) on T-LESS objects (30 classes)
4. Tests detection quality on unseen images
5. Exports the trained model to ONNX with the tensor names expected by `isaac_ros_rtdetr`
6. Downloads T-LESS CAD meshes and converts them to `.obj` for FoundationPose

## Repository structure

```
tless_detector/
├── requirements.txt             # all pip deps (except torch/torchvision)
├── data/
│   └── download_tless.py        # full dataset (pfcalcul) or meshes only (--meshes-only)
├── scripts/
│   ├── prepare_dataset.py       # BOP PBR → COCO JSON
│   ├── verify_dataset.py        # visual sanity check (JupyterLab)
│   ├── predict.py               # run inference on unseen images
│   ├── export_onnx.py           # ONNX export + tensor name fix
│   └── convert_meshes.py        # T-LESS .ply → centered .obj for FoundationPose
├── configs/
│   └── rtdetr_r50vd_tless.yml   # RT-DETR training config
├── cluster/
│   ├── train.sbatch             # SLURM job script (pfcalcul-specific)
│   └── README.md                # step-by-step cluster guide
└── third_party/
    └── RT-DETR/                 # git submodule (lyuwenyu/RT-DETR)
```

## Pre-trained models

Trained weights are hosted on Hugging Face:
**[maximesabbah/tless_rtdetr](https://huggingface.co/maximesabbah/tless_rtdetr)**

| File | Description |
|---|---|
| `best.pth` | PyTorch checkpoint — best validation mAP (~0.94, trained 72 epochs) |
| `tless_rtdetr.onnx` | ONNX export — ready for ONNXRuntime or TensorRT conversion |

### Download

```bash
pip install huggingface_hub

python3 - <<'EOF'
from huggingface_hub import hf_hub_download

# PyTorch checkpoint
hf_hub_download(
    repo_id="maximesabbah/tless_rtdetr",
    filename="best.pth",
    local_dir="output/rtdetr_r50vd_tless/",
)

# ONNX model
hf_hub_download(
    repo_id="maximesabbah/tless_rtdetr",
    filename="tless_rtdetr.onnx",
    local_dir="output/rtdetr_r50vd_tless/",
)
EOF
```

Or with the CLI:
```bash
huggingface-cli download maximesabbah/tless_rtdetr best.pth         --local-dir output/rtdetr_r50vd_tless/
huggingface-cli download maximesabbah/tless_rtdetr tless_rtdetr.onnx --local-dir output/rtdetr_r50vd_tless/
```

## Training: see `cluster/README.md`

The full step-by-step guide (setup → train → test → export → TensorRT) is in
[cluster/README.md](cluster/README.md).

## Preparing meshes for FoundationPose

FoundationPose needs a centered `.obj` mesh for each T-LESS object you want to track.
Run these two steps on any machine with internet access (no GPU required):

```bash
# 1. Download only the CAD models (~100 MB, not the full 15 GB training set)
python data/download_tless.py --meshes-only --out-dir ./tless_meshes

# 2. Convert .ply → .obj (centered at centroid, as required by FoundationPose)
#    tless_models.zip contains three variants: models_cad/ (default, recommended),
#    models_eval/, and models_reconst/. models_cad/ uses the original CAD geometry.
python scripts/convert_meshes.py \
    --models-dir ./tless_meshes/models_cad \
    --out-dir    ${ISAAC_ROS_WS}/isaac_ros_assets/tless_meshes
```

Both scripts accept `--help` for all options.

## After training: Isaac ROS integration

```bash
# On the Isaac ROS machine, inside the container:
/usr/src/tensorrt/bin/trtexec \
    --onnx=tless_rtdetr.onnx \
    --saveEngine=tless_rtdetr.plan \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --fp16

ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
    launch_fragments:=realsense_mono_rect_depth,foundationpose_tracking \
    mesh_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/tless_meshes/obj_000001.obj \
    score_engine_file_path:=score_trt_engine.plan \
    refine_engine_file_path:=refine_trt_engine.plan \
    rt_detr_engine_file_path:=tless_rtdetr.plan \
    interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/quickstart_interface_specs.json
```

Change `obj_000001.obj` to the T-LESS object you want to track (001 to 030).

## Credits

- [RT-DETR v2](https://github.com/lyuwenyu/RT-DETR) by lyuwenyu
- [T-LESS dataset](https://cmp.felk.cvut.cz/t-less/) by Hodaň et al.
- [BOP Benchmark](https://bop.felk.cvut.cz/)

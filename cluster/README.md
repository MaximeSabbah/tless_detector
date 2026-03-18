# Step-by-step guide: training on pfcalcul (LAAS-CNRS)

This guide walks you through every command, from cloning the repo to
a trained model tested on unseen images and exported to ONNX.

> Commands shown with `$` are run in a terminal. Lines starting with `#` are comments.

---

## Overview

```
pfcalcul frontal  →  clone repo, create venv, download dataset, prepare annotations
pfcalcul frontal  →  smoke-test the pipeline (--test-only, no GPU needed)
pfcalcul frontal  →  submit training job to SLURM
pfcalcul compute  →  SLURM runs the training (~8-16h on A100, automatic)
pfcalcul frontal  →  test the trained model on unseen images
pfcalcul frontal  →  export trained model to ONNX
Your robot machine→  convert ONNX to TensorRT .plan (inside Isaac ROS)
```

---

## Part 1 — One-time setup on pfcalcul frontal

### 1.1 — Connect to pfcalcul

```bash
ssh YOUR_LOGIN@pfcalcul.laas.fr
```

### 1.2 — Add SLURM to your PATH (one-time)

```bash
echo '[ -d /usr/local/slurm ] && export PATH="/usr/local/slurm/bin:${PATH}"' >> ~/.bashrc
source ~/.bashrc
sbatch --version   # should print: slurm 22.x.x or similar
```

### 1.3 — Clone the repo and initialize the submodule

```bash
cd ~
git clone https://github.com/YOUR_GITHUB_USERNAME/tless_detector.git
cd tless_detector
git submodule update --init --recursive
```

### 1.4 — Create the Python virtual environment

We use [uv](https://github.com/astral-sh/uv) — a fast drop-in replacement for pip.

```bash
cd ~/tless_detector

# Install uv (one-time, installs to ~/.local/bin)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # make uv available in current shell

# Create the venv
uv venv .venv
source .venv/bin/activate

# Install PyTorch — check your CUDA version first:
#   nvidia-smi | grep "CUDA Version"
# Then pick the matching index:
#   CUDA 11.8 → cu118
#   CUDA 12.1 → cu121
#   CUDA 12.4 → cu124
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
uv pip install -r requirements.txt
```

Verify:
```bash
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### 1.5 — Download the T-LESS dataset

> The dataset is ~15 GB. Run this on the frontal — compute nodes have no internet.

```bash
source ~/.venv/bin/activate   # or: source ~/tless_detector/.venv/bin/activate
python data/download_tless.py
```

When done:
```
Dataset ready at: /datasets/tless
```

Verify:
```bash
ls /datasets/tless/
# camera_primesense.json  train_pbr/  README.md
ls /datasets/tless/train_pbr/ | head
# 000000  000001  000002  ...
```

### 1.6 — Download pretrained weights

```bash
mkdir -p ~/tless_detector/weights
wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth \
     -O ~/tless_detector/weights/rtdetr_r50vd_objects365.pth

ls -lh ~/tless_detector/weights/   # should be ~200 MB
```

### 1.7 — Prepare COCO annotations

```bash
cd ~/tless_detector
python scripts/prepare_dataset.py
```

Takes 2-3 minutes. When done:
```
Saved train: 47500 images → data/tless_coco_train.json
Saved val:    2500 images → data/tless_coco_val.json
```

---

## Part 2 — Optional: visually verify the annotations

Before spending GPU hours, check the annotations look correct.

1. Go to pfcalcul JupyterLab: **https://pfcalcul.laas.fr/cuda-94gb/ → JupyterLAB RTX PRO 6000**
2. Open a Terminal and run:

```bash
cd ~/tless_detector
source .venv/bin/activate
python scripts/verify_dataset.py
```

3. In the file browser (left panel) navigate to `~/tless_detector/verify_output/` and
   click any `.jpg` — you should see green bounding boxes around T-LESS objects.

If boxes look wrong, debug `scripts/prepare_dataset.py` before training.

---

## Part 3 — Smoke-test before submitting

This validates the full pipeline (data loading, model build, one validation pass)
without spending GPU time on a full training run.

```bash
cd ~/tless_detector
source .venv/bin/activate

cd third_party/RT-DETR/rtdetrv2_pytorch
python tools/train.py \
    -c ~/tless_detector/configs/rtdetr_r50vd_tless.yml \
    -t ~/tless_detector/weights/rtdetr_r50vd_objects365.pth \
    --test-only \
    -u val_dataloader.num_workers=0
```

or 

```bash
cd ~/tless_detector
source .venv/bin/activate

cd third_party/RT-DETR/rtdetrv2_pytorch
python tools/train.py \
    -c ~/tless_detector/configs/rtdetr_r50vd_tless.yml \
    -t ~/tless_detector/weights/rtdetr_r50vd_objects365.pth \
    --use-amp \
    -u epoches=1 \
    -u train_dataloader.total_batch_size=2

```

Expected: it runs through the val set (slow without GPU — that's normal) and
prints COCO mAP metrics. mAP will be near 0 since the model is not trained yet.
**If it completes without a crash, the pipeline is correct.**

---

## Part 4 — Submit the training job

### 4.1 — Create the logs directory (required before submitting)

SLURM opens the log files before the job starts, so the directory must exist:

```bash
mkdir -p ~/tless_detector/logs
```

### 4.2 — Submit

```bash
sbatch ~/tless_detector/cluster/train.sbatch
# Output: Submitted batch job 12345
```

### 4.3 — Monitor

```bash
# See job status (PD=pending, R=running, CG=completing):
squeue -u YOUR_LOGIN

# Live log output (replace 12345 with your job ID):
tail -f ~/tless_detector/logs/train_12345.out
```

You will see output like:
```
===========================================
Job ID:   12345
Node:     gpu-node-3
GPU:      NVIDIA A100-SXM4-40GB, 40536 MiB
===========================================
Syncing dataset ...
Sync complete.
--- Environment ---
Python 3.12.x
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
-------------------
Epoch [1/72] loss: 12.34 ...
```

Press `Ctrl+C` to stop watching — the job keeps running.

### 4.4 — Check outcome

```bash
# Check exit code after the job finishes:
sacct -j 12345 --format=JobID,State,ExitCode
# State=COMPLETED, ExitCode=0:0 → success
# State=FAILED → check the error log:
cat ~/tless_detector/logs/train_12345.err
```

### 4.5 — Cancel (if needed)

```bash
scancel 12345
```

---

## Part 5 — Test on unseen images

Before exporting, verify the trained model actually detects T-LESS objects on
images not seen during training.

```bash
cd ~/tless_detector
source .venv/bin/activate

python scripts/predict.py \
    --checkpoint output/rtdetr_r50vd_tless/best.pth \
    --images     /path/to/your/test/images/ \
    --output     output/predictions/ \
    --threshold  0.5
```

The script saves annotated images with bounding boxes and confidence scores to
`output/predictions/`. Open them in JupyterLab to visually inspect quality.

If detections look good → proceed to ONNX export.
If detections are poor → check training logs, consider more epochs or lower threshold.

---

## Part 6 — Export to ONNX

```bash
cd ~/tless_detector
source .venv/bin/activate

python scripts/export_onnx.py \
    --checkpoint output/rtdetr_r50vd_tless/best.pth \
    --output     output/tless_rtdetr.onnx
```

Expected output at the end:
```
[VERIFY] ✓ All shape checks passed.
==================================================
Final ONNX saved at: output/tless_rtdetr.onnx
==================================================
```

---

## Part 7 — Copy results to your robot machine

```bash
# From your local machine or robot machine:
scp YOUR_LOGIN@pfcalcul.laas.fr:~/tless_detector/output/tless_rtdetr.onnx \
    ${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/
```

---

## Part 8 — Convert to TensorRT on the Isaac ROS machine

Run **inside the Isaac ROS Docker container**:

```bash
mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/

/usr/src/tensorrt/bin/trtexec \
    --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/tless_rtdetr.onnx \
    --saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/tless_rtdetr.plan \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --fp16
```

When done: `&&&& PASSED TensorRT.trtexec ...`

---

## Part 9 — Launch FoundationPose

```bash
# Inside the Isaac ROS container:
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
    launch_fragments:=realsense_mono_rect_depth,foundationpose_tracking \
    mesh_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/tless_meshes/obj_000001.obj \
    score_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/score_trt_engine.plan \
    refine_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan \
    rt_detr_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/tless_rtdetr.plan \
    interface_specs_file:=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_foundationpose/quickstart_interface_specs.json
```

Change `obj_000001.obj` to the T-LESS object you want to track (001 to 030).

---

## Troubleshooting

**Job stays PD (pending) for a long time**
Normal — waiting for a GPU to become free.
Check estimated start time: `squeue -u YOUR_LOGIN --start`

**"CUDA out of memory" during training**
Reduce `total_batch_size` in `configs/rtdetr_r50vd_tless.yml` from 8 to 4.

**Training crashes at epoch 1**
Check `logs/train_JOBID.err`. Common causes:
- Wrong `img_folder` path in config (must be `/datasets/tless`)
- COCO JSON not found → run `prepare_dataset.py` first
- Submodule not initialized → `git submodule update --init --recursive`

**trtexec fails with "Failed to parse ONNX"**
The ONNX was exported with wrong tensor names. Re-run `export_onnx.py` and
check the `[RENAME]` output.

**Venv not found in the SLURM job**
Make sure `~/tless_detector/.venv/` exists on the cluster before submitting.
The `.venv` is local to the machine where you created it.

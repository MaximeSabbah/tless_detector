# Step-by-step guide: training on pfcalcul (LAAS-CNRS)

This guide walks you through every command, from cloning the repo to
getting a trained `.onnx` file ready for Isaac ROS.

> **Tip**: commands shown with `$` are run in a terminal.
> Lines starting with `#` are comments — you don't type those.

---

## Overview of what we are going to do

```
Your laptop        →  build Apptainer container, push repo to GitHub
pfcalcul frontal   →  clone repo, download dataset, prepare annotations
pfcalcul frontal   →  submit training job to SLURM
pfcalcul compute   →  SLURM runs the training (automatic, you just wait)
pfcalcul frontal   →  export trained model to ONNX
Your robot machine →  convert ONNX to TensorRT .plan (inside Isaac ROS)
```

---

## Part 1 — One-time setup on your LOCAL machine

### 1.1 — Edit the placeholders in the repo

Before doing anything else, open every file listed below and replace:
- `YOUR_LOGIN` → your pfcalcul username (e.g. `jdupont`)
- `YOUR_TEAM` → your LAAS team account name (ask your supervisor if unsure)
- `YOUR_NAME` → your name for container labels

Files to edit:
```
configs/rtdetr_r50vd_tless.yml
scripts/prepare_dataset.py
scripts/verify_dataset.py
scripts/export_onnx.py
cluster/tless_detector.def
cluster/train.sbatch
data/download_tless.py
```

### 1.2 — Initialize the RT-DETR submodule

```bash
# In your repo directory:
git submodule update --init --recursive
```

This downloads the official RT-DETR code into `third_party/RT-DETR/`.
You should see `third_party/RT-DETR/rtdetr_pytorch/` appear.

### 1.3 — Build the Apptainer container

This requires `sudo` and `apptainer` installed on your laptop.
If you don't have `apptainer` installed:

```bash
# Ubuntu/Debian:
sudo apt-get install -y apptainer
# or follow: https://apptainer.org/docs/user/main/quick_start.html
```

Then build:

```bash
# This takes 5-15 minutes depending on your internet speed.
# It downloads ~8 GB from NVIDIA NGC.
sudo apptainer build cluster/tless_detector.sif cluster/tless_detector.def
```

You will see a lot of output. It's done when you see:
```
INFO:    Creating SIF file...
INFO:    Build complete: cluster/tless_detector.sif
```

### 1.4 — Push your repo to GitHub

```bash
git add .
git commit -m "Initial repository"
git push
```

### 1.5 — Copy the container to pfcalcul

The `.sif` file is several GB, so copy it separately (not via git):

```bash
# Replace YOUR_LOGIN with your pfcalcul login
scp cluster/tless_detector.sif YOUR_LOGIN@pfcalcul.laas.fr:~/containers/

# If ~/containers/ doesn't exist yet, create it first:
# ssh YOUR_LOGIN@pfcalcul.laas.fr "mkdir -p ~/containers"
```

---

## Part 2 — Setup on pfcalcul frontal

### 2.1 — Connect to pfcalcul

```bash
ssh YOUR_LOGIN@pfcalcul.laas.fr
```

### 2.2 — Add SLURM to your PATH (one-time setup)

```bash
# Open your ~/.bashrc file:
nano ~/.bashrc

# Add this line at the end:
[ -d /usr/local/slurm ] && export PATH="/usr/local/slurm/bin:${PATH}"

# Save (Ctrl+O, Enter, Ctrl+X) and reload:
source ~/.bashrc

# Test that SLURM is accessible:
sbatch --version
# Should print something like: slurm 22.x.x
```

### 2.3 — Clone your repo on pfcalcul

```bash
cd ~
git clone https://github.com/YOUR_GITHUB_USERNAME/tless_detector.git
cd tless_detector

# Initialize the RT-DETR submodule here too:
git submodule update --init --recursive
```

### 2.4 — Download the T-LESS dataset

> ⚠️ **Do this on the frontal, not inside a job.**
> Compute nodes often have no internet access.
> This will take ~20 minutes (downloading ~15 GB).

```bash
cd ~/tless_detector

# Install huggingface_hub if needed (just for the download script)
pip install --user huggingface_hub

python data/download_tless.py
```

You should see progress bars. When done:
```
Dataset ready at: /datasets/tless
```

Verify the download:
```bash
ls /datasets/tless/
# Should show: camera_primesense.json  train_pbr/  README.md
ls /datasets/tless/train_pbr/ | head
# Should show: 000000  000001  000002  ...
```

### 2.5 — Download pretrained weights

```bash
mkdir -p ~/tless_detector/weights
wget https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth \
     -O ~/tless_detector/weights/rtdetr_r50vd_objects365.pth

# Check it downloaded correctly (should be ~200 MB):
ls -lh ~/tless_detector/weights/
```

### 2.6 — Prepare COCO annotations

```bash
cd ~/tless_detector
python scripts/prepare_dataset.py
```

This takes 2-3 minutes. When done:
```
Saved train: 47500 images, ~250000 annotations → /home/YOUR_LOGIN/tless_detector/data/tless_coco_train.json
Saved val:    2500 images,  ~13000 annotations → /home/YOUR_LOGIN/tless_detector/data/tless_coco_val.json
```

---

## Part 3 — Optional: visual check in JupyterLab

Before spending GPU hours training, verify the annotations look correct.

1. Go to the pfcalcul JupyterLab:
   **https://pfcalcul.laas.fr → JupyterLAB RTX PRO 6000**

2. Open a terminal (click the `+` tab, then "Terminal")

3. Run:
   ```bash
   cd ~/tless_detector
   python scripts/verify_dataset.py
   ```

4. In the JupyterLab file browser (left panel), navigate to:
   `~/tless_detector/verify_output/`
   Click on any `.jpg` to preview — you should see green bounding boxes
   around T-LESS objects.

If the boxes look wrong (off-center, wrong objects), stop and debug
`scripts/prepare_dataset.py` before training.

---

## Part 4 — Submit the training job

### 4.1 — Dry-run validation (no GPU used)

```bash
cd ~/tless_detector

# This checks the script syntax and estimates when the job would start,
# but does NOT actually submit anything:
sbatch --test-only cluster/train.sbatch
```

Expected output:
```
Job 12345 to start at 2024-03-01T14:00:00 ...
```

If you get an error, read it carefully — it usually means a path is wrong
or YOUR_LOGIN/YOUR_TEAM hasn't been replaced.

### 4.2 — Submit the real job

```bash
sbatch cluster/train.sbatch
```

Output:
```
Submitted batch job 12345
```

Note your job ID (12345 in this example).

### 4.3 — Monitor the job

```bash
# See all your running/pending jobs:
squeue -u YOUR_LOGIN

# Output columns: JOBID  PARTITION  NAME  USER  STATE  TIME  NODES  NODELIST
# STATE will be:
#   PD = pending (waiting for a GPU to become available)
#   R  = running
#   CG = completing

# Watch the live log output (replace 12345 with your job ID):
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
--- Container environment ---
Python 3.10.x
PyTorch: 2.2.0
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
-----------------------------
Epoch [1/72] loss: 12.34 ...
```

Press `Ctrl+C` to stop watching (the job keeps running).

### 4.4 — Check job status

```bash
# If the job is no longer in squeue, it finished (or failed).
# Check the exit code:
sacct -j 12345 --format=JobID,State,ExitCode

# If State=COMPLETED and ExitCode=0:0 → success!
# If State=FAILED → check the error log:
cat ~/tless_detector/logs/train_12345.err
```

### 4.5 — Cancel a job (if needed)

```bash
scancel 12345
```

---

## Part 5 — Export the trained model to ONNX

Once training is complete (you get an email or check sacct), run:

```bash
cd ~/tless_detector

# Run export inside the Apptainer container (needs GPU for ONNX verification)
apptainer exec \
    --nvcli \
    --bind /home/YOUR_LOGIN:/home/YOUR_LOGIN \
    --bind /datasets:/datasets \
    ~/containers/tless_detector.sif \
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

## Part 6 — Copy results to your robot machine

```bash
# From your LOCAL machine (or robot machine):
scp YOUR_LOGIN@pfcalcul.laas.fr:~/tless_detector/output/tless_rtdetr.onnx \
    ${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/
```

---

## Part 7 — Convert to TensorRT on the Isaac ROS machine

This must be done **inside the Isaac ROS Docker container** on your
robot/workstation (not on pfcalcul).

```bash
# Inside the Isaac ROS container:
mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/

/usr/src/tensorrt/bin/trtexec \
    --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/tless_rtdetr.onnx \
    --saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/tless/tless_rtdetr.plan \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --fp16
```

This takes 5-15 minutes. When done, you will see:
```
&&&& PASSED TensorRT.trtexec ...
```

---

## Part 8 — Launch FoundationPose with your detector

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

### "No space left on device" during download
The `/datasets/` partition may be full. Contact the pfcalcul team.

### Job stays in PD (pending) for a long time
Normal — you are waiting for a GPU to become free.
Check estimated start time: `squeue -u YOUR_LOGIN --start`

### "CUDA out of memory" during training
Reduce `batch_size` in `configs/rtdetr_r50vd_tless.yml` from 8 to 4.

### Training crashes at epoch 1
Check `logs/train_JOBID.err`. Common causes:
- Wrong `img_folder` path in config (must be `/datasets/tless`)
- COCO JSON not found (run `prepare_dataset.py` first)
- Submodule not initialized (`git submodule update --init --recursive`)

### trtexec fails with "Failed to parse ONNX"
The ONNX was exported with wrong tensor names.
Re-run `export_onnx.py` and check the `[RENAME]` output.

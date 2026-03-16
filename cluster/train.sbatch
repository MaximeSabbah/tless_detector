#!/bin/bash
# =============================================================================
# SLURM job script for T-LESS RT-DETR training on pfcalcul (LAAS-CNRS)
#
# Submit with:   sbatch cluster/train.sbatch
# Monitor with:  squeue -u YOUR_LOGIN
# Logs at:       ~/tless_detector/logs/train_JOBID.out
#
# IMPORTANT: Replace YOUR_LOGIN and YOUR_TEAM before submitting.
# =============================================================================

#SBATCH --job-name=tless_rtdetr
#SBATCH --partition=gpu
#SBATCH --account=team_YOUR_TEAM      # Ask your team for the correct account name
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # 8 CPU cores (4 for dataloader + headroom)
#SBATCH --mem=32G                     # 32 GB RAM
#SBATCH --time=20:00:00               # 20 hours max (72 epochs ~ 8-16h)
#SBATCH --output=/home/YOUR_LOGIN/tless_detector/logs/train_%j.out
#SBATCH --error=/home/YOUR_LOGIN/tless_detector/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_LOGIN@laas.fr

# =============================================================================
# Paths — only edit if you changed the repo location
# =============================================================================
REPO=/home/YOUR_LOGIN/tless_detector
CONTAINER=/home/YOUR_LOGIN/containers/tless_detector.sif

# =============================================================================
# Pre-flight checks
# =============================================================================
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    echo "Build it locally and copy with:"
    echo "  sudo apptainer build cluster/tless_detector.sif cluster/tless_detector.def"
    echo "  scp cluster/tless_detector.sif YOUR_LOGIN@pfcalcul.laas.fr:~/containers/"
    exit 1
fi

if [ ! -f "${REPO}/data/tless_coco_train.json" ]; then
    echo "ERROR: COCO annotations not found at ${REPO}/data/tless_coco_train.json"
    echo "Run on the frontal first: python scripts/prepare_dataset.py"
    exit 1
fi

if [ ! -f "${REPO}/weights/rtdetr_r50vd_objects365.pth" ]; then
    echo "ERROR: Pretrained weights not found at ${REPO}/weights/rtdetr_r50vd_objects365.pth"
    echo "Download with:"
    echo "  mkdir -p ${REPO}/weights"
    echo "  wget https://github.com/lyuwenyu/RT-DETR/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth \\"
    echo "       -O ${REPO}/weights/rtdetr_r50vd_objects365.pth"
    exit 1
fi

mkdir -p ${REPO}/logs

# =============================================================================
# Job info
# =============================================================================
echo "==========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Start:    $(date)"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==========================================="

# =============================================================================
# Sync dataset to the compute node's fast local storage.
# pfcalcul-specific: /pfcalcul/tools/datasync copies /datasets/tless/ from the
# frontal's NFS to this node's local /datasets/ directory.
# Path is identical on frontal and compute node — no config changes needed.
# If the data is already synced (previous run), this is near-instant.
# =============================================================================
echo "Syncing dataset to local node storage ..."
/pfcalcul/tools/datasync /datasets/tless/
echo "Sync complete."

# =============================================================================
# Run training inside the Apptainer container.
#
# --nvcli   : pfcalcul-specific GPU passthrough (more efficient than --nv).
#             Documented in the platform's GPU guide.
# --bind    : makes /home and /datasets visible at the same paths inside
#             the container as outside (no path changes needed in configs).
# =============================================================================
echo ""
echo "Starting training ..."
echo ""

apptainer exec \
    --nvcli \
    --bind /home/YOUR_LOGIN:/home/YOUR_LOGIN \
    --bind /datasets:/datasets \
    ${CONTAINER} \
    bash -c "
        set -e

        echo '--- Container environment ---'
        python --version
        python -c 'import torch; print(\"PyTorch:\", torch.__version__)'
        python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'
        python -c 'import torch; print(\"GPU:\", torch.cuda.get_device_name())'
        echo '-----------------------------'
        echo ''

        cd ${REPO}

        python third_party/RT-DETR/rtdetr_pytorch/tools/train.py \
            -c configs/rtdetr_r50vd_tless.yml \
            -r weights/rtdetr_r50vd_objects365.pth \
            --use-amp
    "

EXIT_CODE=$?

echo ""
echo "==========================================="
echo "Job finished at $(date)"
echo "Exit code: ${EXIT_CODE}"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Best checkpoint: ${REPO}/output/rtdetr_r50vd_tless/best.pth"
fi
echo "==========================================="

exit ${EXIT_CODE}

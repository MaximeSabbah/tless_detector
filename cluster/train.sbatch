#!/bin/bash
# =============================================================================
# SLURM job script for T-LESS RT-DETR training on pfcalcul (LAAS-CNRS)
#
# Submit:  sbatch cluster/train.sbatch
# Monitor: squeue -u msabbah
# Logs:    ~/tless_detector/logs/train_JOBID.{out,err}
# =============================================================================

#SBATCH --job-name=tless_rtdetr
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=35gb
#SBATCH --time=20:00:00
#SBATCH --output=/home/msabbah/tless_detector/logs/train_%j.out
#SBATCH --error=/home/msabbah/tless_detector/logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=msabbah@laas.fr

REPO=/home/msabbah/tless_detector

# =============================================================================
# Pre-flight checks
# =============================================================================
if [ ! -f "${REPO}/data/tless_coco_train.json" ]; then
    echo "ERROR: COCO annotations not found."
    echo "       Run first: cd ${REPO} && python scripts/prepare_dataset.py"
    exit 1
fi

if [ ! -f "${REPO}/weights/rtdetr_r50vd_objects365.pth" ]; then
    echo "ERROR: Pretrained weights not found at ${REPO}/weights/"
    exit 1
fi

if [ ! -d "${REPO}/.venv" ]; then
    echo "ERROR: venv not found at ${REPO}/.venv"
    echo "       Run setup first — see cluster/README.md Part 1.4"
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
# Sync dataset to the compute node's fast local storage
# =============================================================================
echo "Syncing dataset to local node storage ..."
/pfcalcul/tools/datasync /datasets/tless/
echo "Sync complete."

echo ""
echo "Starting training ..."
echo ""

source ${REPO}/.venv/bin/activate

echo '--- Environment ---'
python --version
python -c 'import torch; print("PyTorch:", torch.__version__)'
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
python -c 'import torch; print("GPU:", torch.cuda.get_device_name())'
echo '-------------------'
echo ''

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${REPO}/third_party/RT-DETR/rtdetrv2_pytorch
python tools/train.py \
    -c ${REPO}/configs/rtdetr_r50vd_tless.yml \
    -t ${REPO}/weights/rtdetr_r50vd_objects365.pth \
    --use-amp

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

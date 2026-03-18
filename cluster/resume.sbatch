#!/bin/bash
# =============================================================================
# SLURM job script to RESUME T-LESS RT-DETR training on pfcalcul (LAAS-CNRS)
#
# Use this when a previous training job was killed (wall-time, OOM, etc.)
# and you want to continue from the last saved checkpoint.
#
# Submit:  sbatch cluster/resume.sbatch
# Monitor: squeue -u msabbah
# Logs:    ~/tless_detector/logs/resume_JOBID.{out,err}
# =============================================================================

#SBATCH --job-name=tless_resume
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --output=/home/msabbah/tless_detector/logs/resume_%j.out
#SBATCH --error=/home/msabbah/tless_detector/logs/resume_%j.err
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=msabbah@laas.fr

REPO=/home/msabbah/tless_detector
CHECKPOINT=${REPO}/output/rtdetr_r50vd_tless/checkpoint.pth

# =============================================================================
# Pre-flight checks
# =============================================================================
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found at ${CHECKPOINT}"
    echo "       Nothing to resume. Run train.sbatch first."
    exit 1
fi

if [ ! -f "${REPO}/data/tless_coco_train.json" ]; then
    echo "ERROR: COCO annotations not found."
    exit 1
fi

if [ ! -d "${REPO}/.venv" ]; then
    echo "ERROR: venv not found at ${REPO}/.venv"
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
echo "Resuming: ${CHECKPOINT}"
echo "==========================================="

# =============================================================================
# Sync dataset to the compute node's fast local storage
# =============================================================================
echo "Syncing dataset to local node storage ..."
/pfcalcul/tools/datasync /datasets/tless/
echo "Sync complete."

echo ""
echo "Resuming training ..."
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
export PYTHONUNBUFFERED=1

cd ${REPO}/third_party/RT-DETR/rtdetrv2_pytorch
# -r resumes from checkpoint: restores weights + optimizer + epoch counter + LR scheduler
torchrun --master_port=9909 --nproc_per_node=1 tools/train.py \
    -c ${REPO}/configs/rtdetr_r50vd_tless.yml \
    -r ${CHECKPOINT} \
    --use-amp \
    --seed=0 \
    -u print_freq=10

EXIT_CODE=$?

echo ""
echo "==========================================="
echo "Job finished at $(date)"
echo "Exit code: ${EXIT_CODE}"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training complete!"
    echo "Best checkpoint: ${REPO}/output/rtdetr_r50vd_tless/best.pth"
else
    echo "Job did not complete normally (exit code ${EXIT_CODE})."
    echo "If this was a wall-time kill, run: sbatch cluster/resume.sbatch"
fi
echo "==========================================="

exit ${EXIT_CODE}

#!/bin/sh -e
# =============================================================================
# Submit RT-DETR T-LESS training to pfcalcul SLURM
#
# Usage:
#   bash cluster/train.sbatch --gpu a100:1
#   bash cluster/train.sbatch --gpu a30:1
#
# Monitor: squeue -u msabbah
# Logs at: ~/tless_detector/logs/train_JOBID.{out,err}
# =============================================================================

echo "example: $0 --gpu a100:1"

REPO=/home/msabbah/tless_detector
GPU_REQUIRED=true

. /pfcalcul/tools/sbatchHelpers.sh

# =============================================================================
# Pre-flight checks — run on the frontal before submitting
# =============================================================================
if [ ! -f "${REPO}/data/tless_coco_train.json" ]; then
    echo "ERROR: COCO annotations not found."
    echo "       Run first: cd ${REPO} && python scripts/prepare_dataset.py"
    exit 1
fi

if [ ! -f "${REPO}/weights/rtdetr_r50vd_objects365.pth" ]; then
    echo "ERROR: Pretrained weights not found at ${REPO}/weights/"
    echo "       wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth \\"
    echo "            -O ${REPO}/weights/rtdetr_r50vd_objects365.pth"
    exit 1
fi

if [ ! -d "${REPO}/.venv" ]; then
    echo "ERROR: venv not found at ${REPO}/.venv"
    echo "       Run setup first — see cluster/README.md Part 1.4"
    exit 1
fi

mkdir -p ${REPO}/logs

# =============================================================================
# Submit
# =============================================================================
jobMessage=$(
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name=tless_rtdetr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:${gpu}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=${REPO}/logs/train_%j.out
#SBATCH --error=${REPO}/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=msabbah@laas.fr

echo "==========================================="
echo "Job ID:   \${SLURM_JOB_ID}"
echo "Node:     \$(hostname)"
echo "Start:    \$(date)"
echo "GPU:      \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==========================================="

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

cd ${REPO}/third_party/RT-DETR/rtdetrv2_pytorch
python tools/train.py \
    -c ${REPO}/configs/rtdetr_r50vd_tless.yml \
    -t ${REPO}/weights/rtdetr_r50vd_objects365.pth \
    --use-amp

EXIT_CODE=\$?

echo ""
echo "==========================================="
echo "Job finished at \$(date)"
echo "Exit code: \${EXIT_CODE}"
if [ \${EXIT_CODE} -eq 0 ]; then
    echo "Best checkpoint: ${REPO}/output/rtdetr_r50vd_tless/best.pth"
fi
echo "==========================================="

exit \${EXIT_CODE}
eof
)

showSubmitted

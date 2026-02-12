#!/bin/bash
# ============================================================
# BERT Fine-tuning Job Submission Script
# Run `myinfo` to get your partition, account, and QOS values
# ============================================================
#SBATCH --partition=<YOUR_PARTITION>    # e.g. student
#SBATCH --account=<YOUR_ACCOUNT>        # e.g. is469
#SBATCH --qos=<YOUR_QOS>               # e.g. studentqos
#SBATCH --job-name=bert-finetune-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Load modules
module purge
module load Python/3.11.11-GCCcore-13.3.0
module load CUDA/12.4.0

# Setup venv (first run creates it)
VENV=~/is469-venv
if [ ! -d "$VENV" ]; then
    echo "Creating venv..."
    python3 -m venv $VENV
fi

source $VENV/bin/activate

# Install deps (pip caches so subsequent runs are fast)
pip install -q torch transformers datasets numpy scikit-learn accelerate

# Run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
srun python3 "$SCRIPT_DIR/test_finetune.py"

echo ""
echo "=== Job finished at $(date) ==="

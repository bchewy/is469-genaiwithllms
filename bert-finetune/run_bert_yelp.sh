#!/bin/bash
# ============================================================
# BERT Fine-tuning on Yelp Review Full (5-class)
# Non-interactive sbatch job submission
# ============================================================
#SBATCH --partition=student
#SBATCH --account=is469
#SBATCH --qos=studentqos
#SBATCH --job-name=bert-yelp-finetune
#SBATCH --gres=gpu:1
#SBATCH --constraint="3090|a40|l40s|a100|l40"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=23:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=END,FAIL

echo "============================================================"
echo "BERT Yelp Fine-tuning Job"
echo "============================================================"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Load modules
module purge
module load Python/3.11.11-GCCcore-13.3.0
module load CUDA/12.4.0

# Activate venv
VENV=~/is469-venv
if [ ! -d "$VENV" ]; then
    echo "Creating venv..."
    python3 -m venv $VENV
fi
source $VENV/bin/activate

# Install/update dependencies (pip caches, so fast on subsequent runs)
pip install -q numpy scikit-learn accelerate datasets transformers

# Run training script
echo ""
echo "Starting training..."
echo ""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
srun python3 "$SCRIPT_DIR/bert_finetune_yelp.py"

echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"

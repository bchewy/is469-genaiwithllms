# Selecting GPUs

Two methods for requesting specific GPU models in batch jobs.

## Method 1: Mandatory GPU Selection

Use `--constraint` when a particular GPU model is required. The job stays queued until the specified GPU is available.

```bash
#SBATCH --constraint=a40
```

## Method 2: Optional GPU Selection

Use `--prefer` when GPU preference is flexible. Alternative resources are assigned if the preferred GPU is unavailable.

```bash
#SBATCH --prefer=a40
```

## Using Operators

Combine GPU types with logical operators:
- `|` (OR) -- request either GPU type
- `&` (AND) -- require multiple conditions

Example requesting H100 or H100 NVL:
```bash
srun -p researchlong -c 4 --mem=8gb --gres=gpu:1 --constraint="h100|h100nvl" nvidia-smi
```

## Tags and Resource Selection

A spreadsheet (accessible via SMU credentials) lists available resources and associated tags. The `nopreempt` tag restricts jobs to non-preemptable nodes (limited availability).

### Nopreempt Examples

```bash
# Non-preemptive node only
#SBATCH --constraint="nopreempt"

# Non-preemptive with L40S GPUs
#SBATCH --constraint="nopreempt&l40s"

# Non-preemptive with L40S or V100
#SBATCH --constraint="nopreempt&v100|l40s"
```

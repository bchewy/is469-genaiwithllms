# Job Submission (pip3)

> **Note:** The `researchlong` queue preempts jobs when resources are insufficient. Add `#SBATCH --requeue` to your sbatch file for automatic resubmission.

## Converting Jupyter Notebooks

Convert `.ipynb` files to `.py` using either:
- Jupyter interface: File > Download As > Python
- Command line: `jupyter nbconvert --to script <NOTEBOOK NAME>.ipynb`

## Prerequisites

1. SSH access to the cluster
2. Project files transferred to the cluster
3. Downloaded the pip3 job submission shell script template

## Obtaining Account Info

1. Log into the GPU cluster via SSH
2. Run `myinfo`
3. Record: Account name, Assigned Partition, Assigned QOS

## Template Amendments

| Line | Parameter | Value |
|------|-----------|-------|
| 26 | `--partition` | Your assigned partition |
| 27 | `--account` | Your account name |
| 28 | `--qos` | Your QOS value |
| 29 | `--mail-user` | Your email(s) |
| 30 | `--job-name` | Your job title |

## Module Selection

**For TensorFlow:**
```bash
module purge
module load Python/3.11.7
module load cuDNN/8.9.7.29-CUDA-12.3.2
```

**For PyTorch:**
```bash
module purge
module load Python/3.11.7
module load CUDA/12.4.0
```

## Virtual Environment Setup

Create (first run only):
```bash
python3.11 -m venv ~/myenv
```

Activate before each use:
```bash
source ~/myenv/bin/activate
```

## Package Installation

```bash
pip3 install numpy
pip3 install scikit
```

## Script Execution

```bash
srun --gres=gpu:1 python3 <file path>/myScript.py
```

## Submitting

```bash
chmod +x sbatchTemplatePython.sh
sbatch sbatchTemplatePython.sh
```

Output file format: `<USERNAME>.<JOBID>.out`

```bash
cat IS000G3.1334.out
```

Email notifications are sent when jobs start, complete, end, or fail.

## Useful Commands

| Command | Description |
|---------|-------------|
| `myqueue` | Check job status |
| `myjob <jobid>` | View job details |
| `mypastjob <days>` | View job history (max 30 days) |

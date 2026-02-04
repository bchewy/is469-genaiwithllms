# Job Submission (Conda)

> **Warning:** Avoid running `conda init` permanently. Instead, after loading Anaconda3 via `module load Anaconda3`, execute `eval "$(conda shell.bash hook)"`.

> **Note:** The `researchlong` queue preempts jobs when resources are insufficient. Add `#SBATCH --requeue` to auto-resubmit preempted jobs.

## Converting Jupyter Notebooks

Convert `.ipynb` files to `.py` using either:
- Jupyter interface: File > Download As > Python
- Command line: `jupyter nbconvert --to script <NOTEBOOK NAME>.ipynb`

## Prerequisites

1. SSH access to the cluster established
2. Project files transferred to the cluster
3. Downloaded the conda job submission shell script template

## Obtaining Account Info

1. Log into the GPU cluster
2. Run `myinfo`
3. Record: Account name, Assigned Partition, Assigned QOS

## Modifying the Template

| Line | Parameter | Example |
|------|-----------|---------|
| 26 | `--partition` | `#SBATCH --partition=tester` |
| 27 | `--account` | `#SBATCH --account=is000` |
| 28 | `--qos` | `#SBATCH --qos=is000qos` |
| 29 | `--mail-user` | `#SBATCH --mail-user=user@scis.smu.edu.sg` |
| 30 | `--job-name` | `#SBATCH --job-name=YourName` |

## Loading Modules

```bash
module purge
module load Anaconda3/2022.05
```

## Virtual Environment Setup

Create (first run only):
```bash
conda create -n myenvnamehere
```

Activate (every run):
```bash
conda activate myenvnamehere
```

Install packages:
```bash
conda install pytorch torchvision torchaudio -c pytorch
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

Output appears as `<USERNAME>.<JOBID>.out` in the execution directory.

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

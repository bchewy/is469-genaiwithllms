# Submit Jobs

Users execute workloads through the cluster's workload scheduler, with submission limits based on assigned quotas (viewable via `myinfo`).

## Method 1: `sbatch` (Recommended)

The `sbatch` command executes shell scripts (.sh files) when cluster resources become available, with optional email notifications.

> **Important:** Files written to `/tmp/` may be inaccessible after job completion. Use scratch directories or home directories instead.

```bash
chmod +x <filepath>/shellscript.sh
sbatch /path/to/sh/file.sh
```

### SBATCH Parameters

| Parameter | Description | Format |
|-----------|-------------|--------|
| `--job-name` | Job identifier (no spaces) | `<jobid>.log` |
| `--partition` | Partition assignment | e.g., `project` |
| `--mail-type` | Notification timing | `ALL`, `BEGIN`, `END`, `FAIL` |
| `--mail-user` | Email address | Email address |
| `--time` | Maximum runtime | `HH:MM:SS` |
| `--nodes` | Number of nodes | Integer |
| `--cpus-per-task` | CPU count | Integer |
| `--mem` | Memory requirement | `(Integer)GB` |
| `--gres` | GPU assignment | `gpu:(Integer)` |
| `--output` | Log file location | File path |

No GPUs are assigned unless explicitly requested via `--gres`.

## Method 2: `srun`

Runs interactively, directing output to terminal. Not recommended for lengthy jobs since disconnection terminates execution. Typically used within sbatch scripts.

```bash
srun --partition=normal --nodes=1 --cpus-per-task=30 --mem=2GB /path/to/script.py
```

## Job Queue Management

| Task | Command |
|------|---------|
| View job status | `myqueue` |
| Cancel specific job | `scancel <JOBID>` |
| Cancel all user jobs | `scancel --me` |
| View running/pending job details | `scontrol show jobid <jobid>` |
| View completed job history (by ID) | `sacct --job=<jobid> --format=JobID,User,Jobname,partition,state,time,start,end,elapsed,AllocTRES%45` |
| View completed job history (by name) | `sacct --name=<jobname> --format=JobID,User,Jobname,partition,state,time,start,end,elapsed,AllocTRES%45` |
| View jobs from past N days | `sacct --starttime $(date -d '1 day ago' +"%Y-%m-%d") --format=JobID,User,Jobname,partition,state,time,start,end,elapsed,AllocTRES%45` |

Job detail commands only retrieve currently pending/running jobs or those completed within 5 minutes.

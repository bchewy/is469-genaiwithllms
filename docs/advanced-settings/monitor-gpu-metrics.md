# Monitoring GPU Metrics

`nvidia-smi` is unavailable on the login node since GPUs aren't present there. Use these methods to monitor GPU utilization during job execution.

## Method 1: Batch Script Monitoring (Recommended)

Add monitoring parameters to your batch script before your main workload:

```bash
source ~/myenv/bin/activate
srun whichgpu
srun --gres=gpu:1 python /path/to/your/python/script.py
```

The output file will contain GPU assignment details, e.g.: "You are allocated NVIDIA GeForce RTX 2080 Ti on mustang... You are using GPU 0"

With the node name and GPU number identified, access the **Grafana dashboard** (accessible via SMU network or VPN) to select the compute node and GPU number, then adjust time parameters to review utilization statistics.

## Method 2: SSH into Compute Node

> **Note:** This approach is discouraged when multiple jobs run simultaneously, as SSH randomly selects a job rather than allowing user selection.

1. Run `myqueue` to list active jobs
2. Locate the node name in the `NODELIST(REASON)` column
3. SSH into the identified node
4. Run `nvidia-smi` to view real-time metrics

```bash
myqueue
ssh <nodename>
nvidia-smi
```

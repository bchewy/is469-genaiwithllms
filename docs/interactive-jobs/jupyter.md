# Jupyter Notebook

The `mynotebook` command provisions a Jupyter notebook on the cluster. Notebooks run for up to **6 hours** before requiring a relaunch. For extended tasks, use [`execnotebook`](../non-interactive-jobs/exec-notebook.md) instead.

## Provisioning Steps

1. Run `mynotebook` on the Origami cluster
2. Provide an SMU email address for start/end notifications
3. Choose between CPU-only or GPU-enabled notebooks
4. Select from TensorFlow, PyTorch, or Python 3.11.7 base installation
5. Establish SSH tunnel:
   ```bash
   ssh -N -vv -L 8924:10.2.1.60:8924 exampleuser@origami.smu.edu.sg
   ```
6. Navigate to the provided localhost URL with authentication token
7. Use `scancel` to release resources when finished

> **Limited Availability:** If no GPUs are currently available, GPU notebook requests will not be fulfilled.

## Installing Packages

Install Python libraries within the notebook by prefixing with `!`:

```python
!pip3 install torch
!pip3 install numpy
```

## Reconnecting

Retrieve stored connection instructions:

```bash
cat juypterReconnectingInstructions.txt
```

## Monitoring Usage

Check elapsed time with:

```bash
myqueue
```

# Containers

The cluster uses `enroot` for container support. Enroot uses the same underlying technologies as Docker but without the isolation.

## Download Container Image

Import Docker images from repositories:

```bash
srun --partition=student --mem=2G --cpus-per-task=4 enroot import docker://tensorflow/tensorflow:latest-gpu
```

This generates a `.sqsh` file in the working directory (e.g., `tensorflow+tensorflow+latest-gpu.sqsh`).

## Create Container

Build a container from the downloaded image:

```bash
srun --partition=student --mem=2G --cpus-per-task=10 enroot create --name tensorflow "tensorflow+tensorflow+latest-gpu.sqsh"
```

Verify with:
```bash
enroot list
```

> **Tip:** Remove the `.sqsh` file afterward to preserve home directory space.

## Using the Container

### Interactive (srun)

```bash
srun --pty --partition=tester --mem=8G --cpus-per-task=4 --gres=gpu:1 enroot start tensorflow bash
```

### Batch Job (sbatch)

Mount the home directory as `/tf` within the container and execute a Python script with GPU allocation using an sbatch script.

## Removing Containers

```bash
enroot remove <container_name>
```

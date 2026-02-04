# Dependencies

Libraries like TensorFlow and PyTorch include tested dependency configurations. Adhering to developer-recommended versions helps minimize GPU compatibility challenges on the cluster.

## TensorFlow

Reference the [Tested Build Configurations](https://www.tensorflow.org/install/source#gpu) for compatibility details.

For TensorFlow 2.17: Python 3.9-3.12, cuDNN 8.9, CUDA 12.3.

**Module configuration** (insert after line 52 in sbatch script):

```bash
module purge
module load Python/3.11.7
module load cuDNN/8.9.7.29-CUDA-12.3.2
```

## PyTorch

See [pytorch.org](https://pytorch.org/) for recommended build configurations.

**Installation:**

```bash
pip3 install torch torchvision torchaudio
```

**Module configuration** (insert after line 52 in sbatch script):

```bash
module purge
module load Python/3.11.7
module load CUDA/12.4.0
```

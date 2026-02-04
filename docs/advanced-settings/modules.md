# Module System

The module system addresses varying software requirements across courses, enabling users to select from multiple software versions (e.g., Python 2 or Python 3). It automatically loads necessary dependencies -- for instance, `module load cuDNN` will also load the appropriate CUDA library.

By default, Python 3 loads automatically upon cluster login.

For PyTorch and TensorFlow module setup, see [Dependencies](dependencies.md).

## Core Commands

| Command | Description |
|---------|-------------|
| `module list` | View loaded modules |
| `module available` | View all available modules |
| `module avail <name>` | Search modules by name (case-sensitive) |
| `module spider <name>` | Check all versions of a specific module |
| `module load <name>` | Load a module |
| `module unload <name>` | Unload a module |
| `module purge` | Clear all modules |

## Module Collections

| Command | Description |
|---------|-------------|
| `module save <name>` | Save current modules as a collection |
| `module restore <name>` | Restore modules from a collection |
| `module savelist` | List all saved collections |
| `module describe <name>` | View collection contents |
| `module disable <name>` | Remove a collection |

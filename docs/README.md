# Origami GPU Cluster Documentation

Documentation for the Origami GPU Cluster at SMU, scraped from [violet.scis.dev](https://violet.scis.dev).

The Origami GPU Cluster is a computing resource for submitting intensive computational workloads. GPUs accelerate processing by breaking tasks into smaller components executed in parallel. Users submit jobs through a login node; execution occurs immediately or enters a queue based on resource availability.

**Requirements:** SMU Network access or VPN connectivity.

**Cluster resources:** [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1LmPORRiEdY3rmqNtBJvg0K2S78b-zLgB7GSX95JYhFI/edit?usp=sharing) (requires SMU credentials)

---

## Table of Contents

### Getting Started
1. [Logging In](getting-started/logging-in.md)
2. [Partitions](getting-started/partitions.md)
3. [Starter Jobs](getting-started/starter-jobs.md)
4. [Job Logs](getting-started/job-logs.md)
5. [Transfer Files](getting-started/transfer-files.md)

### Advanced Settings
1. [Module System](advanced-settings/modules.md)
2. [Dependencies](advanced-settings/dependencies.md)
3. [Job Submission (Conda)](advanced-settings/conda-job-submission.md)
4. [Job Submission (pip3)](advanced-settings/pip3-job-submission.md)
5. [Containers](advanced-settings/containers.md)
6. [Selecting GPUs](advanced-settings/select-gpus.md)
7. [Monitoring GPU Metrics](advanced-settings/monitor-gpu-metrics.md)

### Interactive Jobs
1. [Interactive Console](interactive-jobs/interactive-console.md)
2. [Jupyter Notebook](interactive-jobs/jupyter.md)

### Non-Interactive Jobs
1. [Submit Jobs](non-interactive-jobs/submit-jobs.md)
2. [Execute Notebook](non-interactive-jobs/exec-notebook.md)

### Reference
- [Shell Script Generator](shell-script-generator.md)
- [FAQ](faq.md)
- [Changelog](changelog.md)

# Shell Script Generator

A form-based tool on [violet.scis.dev/docs/generateshell](https://violet.scis.dev/docs/generateshell) for generating shell scripts for job submission.

> **Note:** There are no input checks -- if the input for CPU is 99 cores, a script requesting 99 cores will be generated. Check your assigned resources with `myinfo`.

## Configurable Parameters

| Parameter | Description |
|-----------|-------------|
| Environment | conda or pip3 |
| Partition | Your assigned partition |
| Account | Your assigned account |
| QOS | Your assigned QOS level |
| Nodes | Number of nodes requested |
| CPUs | Number of CPU cores |
| Memory | Amount in gigabytes |
| GPUs | Number of GPUs |
| Job Duration | Time limit (days, hours, minutes) |
| Email | Notification recipient |
| Notification Triggers | Begin, End, or Fail events |
| Job Name | Custom job identifier |
| Modules | Pre-loaded modules (e.g., Python/3.11.7) |
| Virtual Environment | Name and whether it already exists |
| Packages | Required packages for execution |

For Jupyter notebook submissions, reference the [conda job submission](advanced-settings/conda-job-submission.md) documentation for module and package specifications.

The form generates a downloadable shell script suitable for cluster job submission.

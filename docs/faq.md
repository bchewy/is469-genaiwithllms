# FAQ

## Command Line Is Slow

When there is a high amount of IO operations on the cluster, the system responds slower. Operations like file listing or copying may take longer during peak usage.

## Login and Access Issues

### General Login Problems

Verify these prerequisites:
- ClearPass installed with Healthy Status
- Connected to WLAN-SMU WiFi network
- No external VPN services active

### Remote Access

To access the GPU cluster outside the SMU network, connect to the SMU VPN (Cisco). Contact your instructor if you don't have VPN credentials.

### Initial Password Change

First-time users are prompted to change their password. Enter the temporary password provided by your instructor when prompted for the "old" password.

## GPU Detection Problems

GPUs only become available when scripts execute through the job scheduler. See the job submission guides for proper execution procedures.

## Job Execution Troubleshooting

### Jobs Not Running

Run `squeue --me` to check job status. A state of `PD` indicates queuing due to resource constraints.

### Job Failures

Common causes:
- Resource competition from other cluster jobs
- Inaccessible file paths in scripts
- Missing Python library installations
- Non-executable template files
- Unloaded required modules (TensorFlow/PyTorch)

## Additional Support

Post questions on the GitHub forum using the format: `[AccountName] <Issue Description>` with relevant details and `.out` files.

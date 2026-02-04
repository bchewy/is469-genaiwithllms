# Starter Jobs

A walkthrough for submitting your first job using a tensor addition sample.

## Tensor Addition Sample

This sample measures the time needed to complete 1 million tensor additions on a 2D matrix, comparing GPU vs CPU performance.

Included files:
- `Add2D.py` -- the executable job code
- `sbatchAdd2D.sh` -- shell script for job submission

## Configuration

Edit the sbatch script (lines 29-32) to insert:
- Assigned partition name
- Account credentials (typically `student`)
- QOS designation
- Email address for notifications

Use the `myinfo` command to retrieve these values.

## File Transfer

Upload files to the cluster using SCP:

```bash
scp <local_filepath> <username>@origami.smu.edu.sg:~
```

## Job Submission

```bash
chmod +x sbatchAdd2D.sh
sbatch sbatchAdd2D.sh
```

## Results

After completion, retrieve the output log via SCP and review performance metrics comparing CPU versus GPU execution times.

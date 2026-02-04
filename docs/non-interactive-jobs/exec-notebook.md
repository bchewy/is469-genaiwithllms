# Execute Notebook

The `execnotebook` command runs Jupyter notebooks for up to **24 hours** without maintaining an active connection. Checkpoint your work to prevent data loss.

## Prerequisites

- Must have previously run `mynotebook`
- The `.ipynb` file must not be stored in the scratch directory (must be in home directory)

## Constraints

- Maximum runtime: 24 hours
- Do not access or modify the notebook file until execution completes

## Usage

1. Run `execnotebook` on the cluster
2. Provide the notebook filename
3. Enter your SMU email address
4. Select framework: TensorFlow, PyTorch, or Python 3.11.7
5. Choose whether to overwrite the existing notebook
6. Specify GPU requirements
7. Submit the job
8. Monitor via `myqueue`

## Output File Naming

When overwriting notebooks: `nameofnotebook.output.ddmmyyyyHHMMSS.ipynb`

## Troubleshooting

If your job remains pending, run `myinfo` to check your concurrent job entitlements.

# Transfer Files

Transferring files to and from the Origami GPU Cluster. You must have a healthy ClearPass installation and SMU VPN access if working remotely.

## Git (Recommended)

Host your project on a git repository and use `git clone` to place the project into your home directory.

## WinSCP (Windows Only)

WinSCP provides a graphical interface for file transfer:

1. Launch the application, enter username/password, click login
2. Accept the host key cache prompt ("Yes")
3. View your home directory in the right pane
4. Navigate local folders using the dropdown in the top left
5. Drag files from the left pane to the right pane
6. Confirm the destination and click "OK"

## SCP (Secure Copy)

### Local to Cluster

```bash
# Single file
scp /path/to/file <username>@origami.smu.edu.sg:~/path/to/destination

# Folder
scp -r /path/to/folder <username>@origami.smu.edu.sg:~/path/to/destination
```

### Cluster to Local

```bash
# Single file
scp <username>@origami.smu.edu.sg:~/path/to/file /path/to/destination

# Folder
scp -r <username>@origami.smu.edu.sg:~/path/to/folder /path/to/destination
```

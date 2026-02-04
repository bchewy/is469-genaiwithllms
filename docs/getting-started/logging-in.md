# Logging In

## Account Access

To use the cluster, obtain an account by reaching out to your instructor with an access request.

## Prerequisites

1. **ClearPass Network Access** -- A functioning ClearPass installation is mandatory. Download from IITS at the support portal.
2. **DNS Configuration** -- Any custom DNS settings (such as `1.1.1.1` or `8.8.8.8`) must be removed from your computer before attempting connection.

## Connection Instructions

Windows users should use PowerShell (avoid Command Prompt).

1. Open Terminal or PowerShell
2. Connect via SSH:
   ```bash
   ssh <accountname>@origami.smu.edu.sg
   ```
3. Confirm host authenticity by typing `yes`
4. Enter the provided password (input remains hidden)
5. Change your password when prompted -- enter current password first, then new password
6. After logout, reconnect using the same SSH command
7. Login with your newly created password

Once connected, you'll access a Bash shell environment.

## Password Management

To change your password while logged in:

1. Run the `passwd` command
2. Type your new password (hidden input)
3. Confirm by entering the password again

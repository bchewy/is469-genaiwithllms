# Interactive Console

The Interactive Console enables a direct connection to the GPU cluster for troubleshooting and interactive work. The session requires continuous connectivity -- disconnection results in session loss and unsaved work disappearing.

## Requesting an Interactive GPU Session

From the login node:

```bash
srun --pty --qos=<QOS-NAME> --partition=<PARTITION-ASSIGNED> --gres=gpu:1 -c=<CORE-COUNT> --mem=<MEMORY-NEEDED> bash
```

> **Note:** There is no need to supply a QOS for prioritized queues.

### Example

```bash
[IS000G3@origami ~]$ srun --pty --qos=research-1-qos --partition=researchshort --gres=gpu:1 -c 10 --mem=64GB bash
[IS000G3@amigo ~]$ nvidia-smi
```

The hostname transition from `origami` to `amigo` confirms successful GPU allocation. `nvidia-smi` verifies GPU assignment.

## Launching Notebooks from Interactive Session

Once connected to a compute node:

```bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=<RANDOM PORT BETWEEN 50000 and 65000>
```

Compute nodes lack external accessibility -- SSH tunneling is mandatory.

### Example

```bash
[IS000G3@amigo ~]$ source notebook-test/bin/activate
(notebook-test) [IS000G3@amigo ~]$ jupyter notebook --no-browser --ip=0.0.0.0 --port=53426
```

Output includes a URL like:
```
http://127.0.0.1:53426/tree?token=370d5288148c34a36e039c38e01b65e8e50d6ccab9a102bd
```

## Establishing SSH Tunnel

From your local machine:

```bash
ssh -N -vv -L <PORT>:<NODENAME>:<PORT> <USERNAME>@origami.smu.edu.sg
```

### Example

```bash
ssh -N -vv -L 53426:amigo:53426 IS000G3@origami.smu.edu.sg
```

Keep the terminal window open throughout your session -- closing it terminates the tunnel.

## Accessing the Notebook

Navigate to the localhost URL from the notebook output, e.g.:
```
http://127.0.0.1:53426/tree?token=370d5288148c34a36e039c38e01b65e8e50d6ccab9a102bd
```

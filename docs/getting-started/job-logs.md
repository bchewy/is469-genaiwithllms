# Job Logs

Job logs capture return values and print values generated during execution. These files are stored in the directory where the `sbatch` command was run.

## Log Structure

### Part 1: Setting Up
Documents module loading and environmental variable initialization.

### Part 2: Standard Output and Error
Captures all job outputs including print statements, return values, and error messages.

### Part 3: Job Report
Summarizes execution outcome, including failure reasons (timeout, code errors, preemption, etc.).

## Common Failure Scenarios

### Erroneous Code

```python
from bs4 import beautifulsoup  # wrong capitalization -- should be BeautifulSoup
```

### Insufficient Memory

```
#SBATCH --mem=1MB
```
Job terminates when memory exhausts.

### Insufficient Time

```
#SBATCH --time=00-00:01:00
```
Job fails with timeout.

## Optimizing Resource Requests

Analyze efficiency metrics from job reports. For example, a job showing 92.78% CPU efficiency and 10.65% memory efficiency means you over-allocated memory.

**Target efficiency:** 80-90% for both CPU and memory.

Adjustments for the example above: request 4-4.5 hours of runtime and 4-4.5 GB of memory instead of the over-allocated 32 GB.

**Benefits:**
- Demonstrates resource stewardship
- Reduces queue wait times for resource-intensive jobs

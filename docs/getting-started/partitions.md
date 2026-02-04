# Partitions

The Origami Cluster features two main categories of partitions (queues) for job submission.

## Community Partitions

| Partition | For |
|-----------|-----|
| **student** | Undergraduate and postgraduate students |
| **project** | UG/PG students working on project assignments |
| **researchlong** | Researchers (up to 5-day max runtime) |
| **researchshort** | Researchers (up to 2-day max runtime) |

## Priority Partitions

| Partition | For |
|-----------|-----|
| **priority** | Research teams who contributed GPU nodes to the cluster |

Community partitions incorporate GPU resources from priority partitions. Users accessing community queues may experience **job preemption** when higher-priority work requires resources.

## Job Preemption

Job preemption occurs when your job is stopped to free up resources for higher-priority tasks. Community queue jobs are stopped first when resources are insufficient. Jobs running on nodes contributed by research teams may be terminated to prioritize that team's research.

## Quality of Service (QOS)

QOS systems enforce resource quotas per account:

| QOS | CPU Cores | RAM | GPU | Max Runtime |
|-----|-----------|-----|-----|-------------|
| student | 4 | 16 GB | 1 | 1 day |
| project | 30 | 30 GB | 1 | 1 day |
| research-1-qos | 10 | 128 GB | 2 | 5 days |
| priority | Unlimited | Unlimited | Any | 5 days |

For additional resources, contact the SCIS IT Team.

Use the `myinfo` command on the server to check your account's partition and QOS assignments.

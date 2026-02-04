# Changelog

## Summer Maintenance Update 2025
*June 19, 2025*

- Upgraded Nvidia Drivers to 570
- Improved NVLink Topology Detection
- Tuned filesystem to improve I/O performance

---

## Summer Maintenance Update 2023
*July 10, 2023 | Maintenance: July 10-17, 2023*

Memory limits are now **strictly enforced**. Jobs exceeding memory usage will be terminated with out-of-memory errors. Adjust memory via `#SBATCH --mem=` in job scripts. Use `myinfo` to view current quotas.

Updates:
- Nvidia drivers upgraded to 535.54.03
- Kernel updated to 4.18.0-477
- Python versions added: 3.8.17, 3.9.17, 3.10.9, 3.11.4
- CUDA versions added: 12.1.1, 12.2.0
- cuDNN versions added: 8.9.2.26, 8.9.3.28
- Transition to cgroupsv2

---

## Winter Maintenance Update 2022
*December 1, 2022 | Maintenance: December 1-16, 2022*

Updates:
- Nvidia drivers upgraded to 525.60.13
- Kernel updated to 4.18.0-425
- Container support added via enroot
- Python versions: 3.7.16, 3.8.16, 3.9.16, 3.10.9
- CUDA versions: 11.6, 11.7, 11.8
- cuDNN version: 8.7.0.84

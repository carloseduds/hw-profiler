# hw-profiler

A lightweight, **portable hardware profiler** that runs from a regular PC/laptop to edge devices.
It collects system information (CPU/RAM/Disk/OS) and, when available, NVIDIA GPU details via:
`nvidia-smi → NVML (pynvml) → torch` (with safe fallbacks).

## Features

- Works on constrained environments (edge/containers) with graceful degradation.
- NVIDIA GPU profiling (when present):
  - VRAM, temperatures, power, utilization, PCIe, compute capability (when available)
  - Derived metrics (when available): memory bandwidth, SM count, CUDA cores, peak FP32, clocks
- Optional topology (`nvidia-smi topo -m`) when supported.
- JSON output for comparisons across machines.
- Built-in **collector health checks** with durations and errors.

## Install

### From source (recommended while developing)

```bash
pip install -e .
```

### Optional dependencies

The profiler can run without these, but will report more when installed:

```bash
pip install psutil pynvml torch
```

## Usage (CLI)

After installation:

```bash
hw-profiler
```

Print JSON:

```bash
hw-profiler --json --pretty
```

Save JSON:

```bash
hw-profiler --out profile.json --pretty
```

Skip topology:

```bash
hw-profiler --no-topology
```

## Usage (Python)

```python
from hw_profiler import build_profile, print_human_summary

profile = build_profile(include_topology=True)
print_human_summary(profile)
```

Quick one-liner style:

```python
from hw_profiler import run_summary

profile = run_summary()
```

## Development

- Code style: PEP 8
- Lint/format suggestions:
  - `ruff`
  - `black`

## License

MIT

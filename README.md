# hw-profiler

A **portable, edge-aware hardware and runtime profiler** for AI/ML workloads.

This project is designed to run **from regular PCs and cloud VMs to constrained edge devices**, providing:
- Robust **hardware capability profiling**
- Optional **runtime profiling** for PyTorch workloads
- Optional **GPU telemetry monitoring** via NVML
- Graceful degradation when dependencies or hardware features are unavailable

## Features

### Core (always available)
- CPU, RAM, disk, OS, container/WSL detection
- NVIDIA GPU detection with safe fallbacks:
  - `nvidia-smi → NVML (pynvml) → torch`
- Derived AI-centric metrics (when available):
  - Memory bandwidth
  - SM count, CUDA cores
  - Base clock
  - Peak FP32 performance
- Multi-GPU support
- Collector health checks (what ran, what failed, duration)

### Optional
- **Runtime profiling** with `torch.profiler`
  - CPU vs CUDA time
  - Top operators (GEMM, attention, etc.)
  - Chrome trace / TensorBoard
- **GPU telemetry monitoring**
  - Utilization, VRAM, power, temperature (time-series)

---

## Installation

### Base (hardware only)
```bash
pip install -e .
````

### Optional extras

```bash
pip install -e ".[torch]"   # runtime profiling
pip install -e ".[nvml]"    # GPU telemetry
pip install -e ".[full]"    # everything
```

---

## Quickstart (Python)

### Hardware profiling

```python
from hw_profiler import run_summary

profile = run_summary()
```

Or manually:

```python
from hw_profiler import build_profile, print_human_summary

profile = build_profile(include_topology=True)
print_human_summary(profile)
```

---

## Pretty JSON (Python)

### Print pretty JSON

```python
import json
from hw_profiler import build_profile

profile = build_profile()
print(json.dumps(profile, indent=2, ensure_ascii=False))
```

### Save pretty JSON

```python
with open("profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2, ensure_ascii=False)
```

---

## CLI Usage

### Human-readable summary

```bash
hw-profiler
```

### Pretty JSON to stdout

```bash
hw-profiler --json --pretty
```

### Save JSON to file

```bash
hw-profiler --out profile.json --pretty
```

### Skip topology (faster / more compatible)

```bash
hw-profiler --no-topology
```

---

## Optional: Runtime Profiling (PyTorch)

> Requires: `pip install -e ".[torch]"`

This mode helps answer:

> **Where is time spent inside model execution?**

### Basic profiling

```python
import torch
from hw_profiler.torch_profiler import profile_model_performance, summarize_top_ops

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.nn.Sequential(
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 4096),
).to(device)

x = torch.randn(16, 4096, device=device)

def step():
    with torch.no_grad():
        _ = model(x)
    # Optional: improves timing accuracy on GPU
    if device == "cuda":
        torch.cuda.synchronize()

result = profile_model_performance(step, num_steps=20)
print(result["events_table"])
```

### Top operators summary

```python
summary = summarize_top_ops(result, top_k=15)
print("\n".join(summary["rows"]))
```

---

## Generate a Chrome Trace (Timeline)

```python
from hw_profiler.torch_profiler import export_chrome_trace

trace_path = export_chrome_trace(step, num_steps=20, out_path="trace.json")
print("Trace saved at:", trace_path)
```

Open `trace.json` in:

* Chrome Trace Viewer: `chrome://tracing`

---

## TensorBoard (Local or Colab)

### Generate TensorBoard logs

```python
import os, shutil
import torch
from torch import profiler

log_root = "tb_logs"
run_name = "run1"

shutil.rmtree(log_root, ignore_errors=True)
os.makedirs(log_root, exist_ok=True)

activities = [profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(profiler.ProfilerActivity.CUDA)

with profiler.profile(
    activities=activities,
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler(log_root, worker_name=run_name),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(20):
        step()
        prof.step()
```

### Open TensorBoard

#### Local

```bash
tensorboard --logdir tb_logs
```

#### Google Colab

```python
%load_ext tensorboard
%tensorboard --logdir tb_logs
```

Go to the **Profile** tab.

---

## Optional: GPU Telemetry Monitor (NVML)

> Requires: `pip install -e ".[nvml]"`

```python
from hw_profiler.gpu_monitor import MonitorConfig, sample_gpu_telemetry

data = sample_gpu_telemetry(
    config=MonitorConfig(duration_s=10, interval_s=0.5),
    best_effort=True,
)

print(data["ok"])
print(len(data.get("samples", [])))
```

---

## Troubleshooting

### TensorBoard shows “No dashboards are active”

* Ensure logs are generated with `tensorboard_trace_handler`
* Ensure logs are inside a run subfolder (`tb_logs/run1/...`)
* Update TensorBoard:

```bash
pip install -U tensorboard
```

### High CPU time from `cudaDeviceSynchronize`

* This is expected when synchronizing for accurate timing
* Remove `torch.cuda.synchronize()` for more realistic pipeline profiling

---

## Project Philosophy

* **Hardware-first**, runtime profiling is optional
* Best-effort collection with clear failure reporting
* Designed for **ML Engineers, MLOps, and Edge AI experimentation**
* Suitable for benchmarking, diagnostics, and technical content creation

---

## License

MIT
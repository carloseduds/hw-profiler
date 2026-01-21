#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hw_profiler

A lightweight, cross-platform hardware profiler designed to run on:
- Regular PCs (Windows/macOS/Linux)
- VMs/containers
- Edge devices (e.g., Raspberry Pi, NVIDIA Jetson)

The profiler prioritizes NVIDIA GPU discovery using:
1) nvidia-smi
2) NVML (pynvml)
3) PyTorch (torch.cuda)

It collects system (CPU/RAM/Disk/OS) information when possible and enriches
GPU results with derived metrics such as:
- Memory bandwidth (GB/s)
- L2 cache size (MB)
- SM count, CUDA cores, base clock, and estimated FP32 peak (GFLOPS)

Notes
-----
- Optional dependencies improve coverage: psutil, pynvml, torch.
- Some GPU fields are not consistently exposed by nvidia-smi across drivers
  and environments. This module uses multiple fallbacks and best-effort
  collection to avoid crashes on constrained edge environments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


def _run_collector(name: str, func, *args, **kwargs) -> tuple[dict, object]:
    """Run a collector and capture status, timing, and errors."""
    start = time.perf_counter()
    status: dict = {"name": name, "attempted": True, "ok": False, "error": None}
    result = None
    try:
        result = func(*args, **kwargs)
        status["ok"] = True
    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
    finally:
        status["duration_ms"] = int((time.perf_counter() - start) * 1000)
    return status, result



def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_import(name: str):
    """Import a module by name, returning None on failure."""
    try:
        return __import__(name)
    except Exception:
        return None


def _run_cmd(cmd: List[str], timeout: int = 8) -> JsonDict:
    """Run a command and return a structured result."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "cmd": cmd,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "cmd": cmd,
        }


def _which(cmd: str) -> bool:
    """Return True if command exists in PATH."""
    return shutil.which(cmd) is not None


def _read_first_line(path: str) -> str:
    """Read the first line of a text file. Returns empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return (file.readline() or "").strip()
    except Exception:
        return ""


def _human_bytes(num_bytes: Optional[float]) -> Optional[str]:
    """Convert a byte value to a human-friendly string."""
    if num_bytes is None:
        return None
    try:
        value = float(num_bytes)
    except Exception:
        return None

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _to_int(value: Any) -> Optional[int]:
    """Best-effort conversion to int."""
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float."""
    try:
        return float(str(value).strip())
    except Exception:
        return None


def get_system_info() -> JsonDict:
    """Collect OS/CPU/memory/disk details (best-effort, cross-platform)."""
    info: JsonDict = {
        "timestamp_utc": _now_iso(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env": {
            "container": bool(os.environ.get("container"))
            or os.path.exists("/.dockerenv"),
            "wsl": "microsoft"
            in (platform.release().lower() + " " + platform.version().lower()),
        },
    }

    psutil = _safe_import("psutil")
    if psutil:
        # Memory
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            info["memory"] = {
                "ram_total_bytes": int(vm.total),
                "ram_total_human": _human_bytes(vm.total),
                "ram_available_bytes": int(vm.available),
                "ram_available_human": _human_bytes(vm.available),
                "swap_total_bytes": int(sm.total),
                "swap_total_human": _human_bytes(sm.total),
            }
        except Exception:
            info["memory"] = {}

        # CPU
        try:
            cpu_freq = None
            try:
                freq = psutil.cpu_freq()
                if freq:
                    cpu_freq = {
                        "current_mhz": freq.current,
                        "min_mhz": freq.min,
                        "max_mhz": freq.max,
                    }
            except Exception:
                cpu_freq = None

            info["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "freq": cpu_freq,
            }
        except Exception:
            info["cpu"] = {}

        # Disks
        try:
            disks: List[JsonDict] = []
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append(
                        {
                            "device": part.device,
                            "mountpoint": part.mountpoint,
                            "fstype": part.fstype,
                            "total_bytes": int(usage.total),
                            "total_human": _human_bytes(usage.total),
                            "free_bytes": int(usage.free),
                            "free_human": _human_bytes(usage.free),
                        }
                    )
                except Exception:
                    continue
            info["disk"] = disks
        except Exception:
            info["disk"] = []
    else:
        info["memory"] = {}
        info["cpu"] = {}
        info["disk"] = []

    # Linux extras (helpful on edge devices)
    if platform.system().lower() == "linux":
        info["linux"] = {
            "proc_cpuinfo_first_line": _read_first_line("/proc/cpuinfo"),
            "proc_meminfo_first_line": _read_first_line("/proc/meminfo"),
        }

        # NVIDIA Jetson (L4T) marker when available
        l4t = _read_first_line("/etc/nv_tegra_release")
        if l4t:
            info["linux"]["nv_tegra_release"] = l4t

    return info


def _parse_nvidia_smi_query(csv_text: str, fields: List[str]) -> List[JsonDict]:
    """Parse 'nvidia-smi --query-gpu' CSV output."""
    gpus: List[JsonDict] = []
    for line in (csv_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            continue
        row: JsonDict = {k: v for k, v in zip(fields, parts)}
        gpus.append(row)
    return gpus


def get_nvidia_smi_basic() -> Optional[JsonDict]:
    """Collect basic NVIDIA GPU info using nvidia-smi, when available."""
    if not _which("nvidia-smi"):
        return None

    fields = [
        "index",
        "name",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "util_gpu_pct",
        "util_mem_pct",
        "pci_bus_id",
        "pcie_gen_current",
        "pcie_gen_max",
        "pcie_width_current",
        "pcie_width_max",
        "compute_cap",
        "cuda_version",
    ]

    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    res = _run_cmd(cmd, timeout=10)

    if res["ok"] and res["stdout"]:
        rows = _parse_nvidia_smi_query(res["stdout"], fields)
        for row in rows:
            # Normalize numeric fields (best-effort)
            for key in list(row.keys()):
                if key.endswith(("_mib", "_c", "_w", "_pct")):
                    row[key] = _to_float(row[key])
                if key in (
                    "index",
                    "pcie_width_current",
                    "pcie_width_max",
                    "pcie_gen_current",
                    "pcie_gen_max",
                ):
                    row[key] = _to_int(row[key])

            total_mib = row.get("memory_total_mib") or 0
            total_bytes = int(float(total_mib) * 1024 * 1024)
            row["memory_total_bytes"] = total_bytes
            row["memory_total_human"] = _human_bytes(total_bytes)

        return {
            "method": "nvidia-smi",
            "gpus": rows,
            "raw": {"cmd": res["cmd"], "stderr": res["stderr"]},
        }

    # Fallback with fewer fields (some drivers reject certain keys)
    fields2 = ["index", "name", "driver_version", "memory.total", "pci.bus_id"]
    cmd2 = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields2)}",
        "--format=csv,noheader,nounits",
    ]
    res2 = _run_cmd(cmd2, timeout=10)
    if res2["ok"] and res2["stdout"]:
        rows2 = _parse_nvidia_smi_query(res2["stdout"], fields2)
        return {
            "method": "nvidia-smi-fallback",
            "gpus": rows2,
            "raw": {"cmd": res2["cmd"], "stderr": res2["stderr"]},
        }

    return {
        "method": "nvidia-smi",
        "error": res["stderr"] or "Failed to execute nvidia-smi.",
    }


def get_nvidia_topology() -> Optional[JsonDict]:
    """Return NVIDIA topology matrix using `nvidia-smi topo -m` when available."""
    if not _which("nvidia-smi"):
        return None

    res = _run_cmd(["nvidia-smi", "topo", "-m"], timeout=10)
    if res["ok"] and res["stdout"]:
        return {"method": "nvidia-smi-topo", "matrix": res["stdout"]}
    return None


def get_nvml_info() -> Optional[JsonDict]:
    """Collect GPU info using NVML (pynvml) when available."""
    try:
        import pynvml  # type: ignore
    except Exception:
        return None

    out: JsonDict = {"method": "pynvml", "gpus": []}

    try:
        pynvml.nvmlInit()

        driver = pynvml.nvmlSystemGetDriverVersion()
        try:
            driver = driver.decode("utf-8", errors="ignore")
        except Exception:
            pass

        count = int(pynvml.nvmlDeviceGetCount())
        out["driver_version"] = driver
        out["count"] = count

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            try:
                name = name.decode("utf-8", errors="ignore")
            except Exception:
                pass

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = None

            power = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = None

            uuid = None
            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                try:
                    uuid = uuid.decode("utf-8", errors="ignore")
                except Exception:
                    pass
            except Exception:
                uuid = None

            out["gpus"].append(
                {
                    "index": i,
                    "name": name,
                    "uuid": uuid,
                    "memory_total_bytes": int(mem.total),
                    "memory_total_human": _human_bytes(mem.total),
                    "memory_used_bytes": int(mem.used),
                    "memory_used_human": _human_bytes(mem.used),
                    "memory_free_bytes": int(mem.free),
                    "memory_free_human": _human_bytes(mem.free),
                    "temperature_c": temp,
                    "power_draw_w": power,
                }
            )

        pynvml.nvmlShutdown()
        return out
    except Exception as exc:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return {"method": "pynvml", "error": str(exc)}


def get_torch_gpu_info() -> Optional[JsonDict]:
    """Collect CUDA device info using PyTorch when available."""
    torch = _safe_import("torch")
    if not torch:
        return None

    out: JsonDict = {"method": "torch", "cuda_available": False, "gpus": []}

    try:
        out["torch_version"] = getattr(torch, "__version__", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["hip_version"] = getattr(torch.version, "hip", None)
    except Exception:
        pass

    if not out["cuda_available"]:
        return out

    try:
        count = int(torch.cuda.device_count())
        out["count"] = count

        for i in range(count):
            props = torch.cuda.get_device_properties(i)

            compute_cap = None
            try:
                cc = torch.cuda.get_device_capability(i)
                compute_cap = f"{cc[0]}.{cc[1]}"
            except Exception:
                compute_cap = None

            total_mem = getattr(props, "total_memory", None)

            out["gpus"].append(
                {
                    "index": i,
                    "name": getattr(props, "name", None),
                    "total_memory_bytes": (
                        int(total_mem) if total_mem is not None else None
                    ),
                    "total_memory_human": _human_bytes(total_mem),
                    "multi_processor_count": getattr(
                        props, "multi_processor_count", None
                    ),
                    "major": getattr(props, "major", None),
                    "minor": getattr(props, "minor", None),
                    "compute_cap": compute_cap,
                }
            )
    except Exception as exc:
        out["error"] = str(exc)

    return out


def compute_ai_features(compute_cap_str: Optional[str]) -> Optional[JsonDict]:
    """Map compute capability to high-level AI hardware features."""
    if not compute_cap_str:
        return None

    match = re.match(r"^\s*(\d+)\.(\d+)\s*$", str(compute_cap_str))
    if not match:
        return None

    major = int(match.group(1))
    minor = int(match.group(2))

    feats: JsonDict = {
        "compute_capability": f"{major}.{minor}",
        "ai_hardware_rating": None,
        "tensor_cores": None,
        "supported_precisions": [],
        "sparsity_support": False,
        "transformer_engine": False,
    }

    # Heuristic mapping based on NVIDIA architecture families.
    if major >= 9:
        feats.update(
            {
                "ai_hardware_rating": "Top AI / Datacenter",
                "tensor_cores": "4th Gen+ Tensor Cores",
                "supported_precisions": [
                    "FP16",
                    "BF16",
                    "TF32",
                    "FP8",
                    "INT8",
                    "sparse",
                ],
                "sparsity_support": True,
                "transformer_engine": True,
            }
        )
    elif major == 8:
        feats.update(
            {
                "ai_hardware_rating": "High-End AI",
                "tensor_cores": "3rd Gen Tensor Cores",
                "supported_precisions": ["FP16", "BF16", "TF32", "INT8", "sparse"],
                "sparsity_support": True,
                "transformer_engine": False,
            }
        )
    elif major == 7 and minor >= 5:
        feats.update(
            {
                "ai_hardware_rating": "Mid-Range AI",
                "tensor_cores": "2nd Gen Tensor Cores",
                "supported_precisions": ["FP16", "INT8", "INT4"],
                "sparsity_support": False,
                "transformer_engine": False,
            }
        )
    elif major >= 7:
        feats.update(
            {
                "ai_hardware_rating": "Entry AI",
                "tensor_cores": "1st Gen Tensor Cores (limited)",
                "supported_precisions": ["FP16"],
                "sparsity_support": False,
                "transformer_engine": False,
            }
        )
    else:
        feats.update(
            {
                "ai_hardware_rating": "Legacy",
                "tensor_cores": None,
                "supported_precisions": ["FP32"],
                "sparsity_support": False,
                "transformer_engine": False,
            }
        )

    return feats


def estimate_llm_capacity(memory_gb: Optional[float]) -> Optional[JsonDict]:
    """Estimate which LLM sizes may fit in VRAM for common quantization modes."""
    if memory_gb is None:
        return None

    def fits(params_b: int, gb_per_b: float, overhead_gb: float = 2.0) -> bool:
        required = params_b * gb_per_b + overhead_gb
        return memory_gb >= required

    tiers: JsonDict = {
        "fp16": {"gb_per_1b": 2.0, "models": {}},
        "int8": {"gb_per_1b": 1.0, "models": {}},
        "int4": {"gb_per_1b": 0.5, "models": {}},
    }

    for precision, cfg in tiers.items():
        gb_per_1b = cfg["gb_per_1b"]
        cfg["models"] = {
            "7B": fits(7, gb_per_1b),
            "13B": fits(13, gb_per_1b),
            "34B": fits(34, gb_per_1b),
            "70B": fits(70, gb_per_1b),
        }

    return tiers


def _version_ge(v1: Any, v2: Any) -> bool:
    """Return True if semantic-ish version v1 >= v2 (best-effort)."""
    try:
        a = [int(x) for x in str(v1).split(".") if x.strip().isdigit()]
        b = [int(x) for x in str(v2).split(".") if x.strip().isdigit()]
        if not a or not b:
            return True
        n = max(len(a), len(b))
        a += [0] * (n - len(a))
        b += [0] * (n - len(b))
        return a >= b
    except Exception:
        return True


def check_software_compatibility(cuda_version: Any, compute_cap: Any) -> JsonDict:
    """Heuristic compatibility matrix for common ML frameworks."""
    compat: JsonDict = {
        "pytorch": {
            "min_cuda": "11.0",
            "recommended_cuda": "12.1",
            "min_compute": "7.0",
        },
        "tensorflow": {
            "min_cuda": "11.2",
            "recommended_cuda": "12.2",
            "min_compute": "7.0",
        },
        "jax": {
            "min_cuda": "11.1",
            "recommended_cuda": "12.1",
            "min_compute": "7.0",
        },
        "huggingface": {
            "min_cuda": "11.0",
            "recommended_cuda": "12.1",
            "min_compute": "7.0",
        },
    }

    out: JsonDict = {
        "cuda_version": cuda_version,
        "compute_cap": compute_cap,
        "frameworks": {},
    }

    cap_ok_default = True
    if compute_cap:
        cap_ok_default = _version_ge(compute_cap, "7.0")

    for fw, req in compat.items():
        cuda_ok = True
        if cuda_version and cuda_version != "Unknown":
            cuda_ok = _version_ge(cuda_version, req["min_cuda"])

        compute_ok = cap_ok_default
        if compute_cap:
            compute_ok = _version_ge(compute_cap, req["min_compute"])

        out["frameworks"][fw] = {
            "supported": bool(cuda_ok and compute_ok),
            "min_cuda": req["min_cuda"],
            "recommended_cuda": req["recommended_cuda"],
            "min_compute": req["min_compute"],
            "cuda_ok": cuda_ok,
            "compute_ok": compute_ok,
        }

    return out


def _cores_per_sm_from_cc(compute_cap: Optional[str]) -> Optional[int]:
    """Return an approximate FP32 cores-per-SM for a given compute capability."""
    if not compute_cap:
        return None

    match = re.match(r"^\s*(\d+)\.(\d+)\s*$", str(compute_cap))
    if not match:
        return None

    major = int(match.group(1))
    minor = int(match.group(2))

    # Practical mapping for common architectures (heuristic).
    # - 7.5 (Turing): 64
    # - 8.0 (A100): 64
    # - 8.6/8.7 (Ampere GA10x): 128
    # - 8.9 (Ada): 128
    # - 9.x (Hopper+): 128 (approx for quick estimates)
    if major == 7 and minor == 5:
        return 64
    if major == 8 and minor == 0:
        return 64
    if major == 8 and minor in (6, 7):
        return 128
    if major == 8 and minor == 9:
        return 128
    if major >= 9:
        return 128
    return None


def _peak_fp32_gflops(
    total_cuda_cores: Optional[int],
    base_clock_mhz: Optional[float],
) -> Optional[float]:
    """Estimate peak FP32 throughput (GFLOPS) using an FMA model."""
    if not total_cuda_cores or not base_clock_mhz:
        return None
    # FP32 FMA ~ 2 ops per cycle per core (heuristic).
    return float(total_cuda_cores) * 2.0 * (float(base_clock_mhz) / 1000.0)


def _memory_bandwidth_gbs(
    bus_width_bits: Optional[int],
    mem_clock_mhz: Optional[float],
) -> Optional[float]:
    """Estimate memory bandwidth (GB/s) from bus width and memory clock."""
    if not bus_width_bits or not mem_clock_mhz:
        return None
    # DDR factor 2; bytes/cycle = bus_width_bits / 8.
    return (float(bus_width_bits) / 8.0) * (float(mem_clock_mhz) * 1e6) * 2.0 / 1e9


def enrich_gpu_with_torch(gpu_block: JsonDict) -> JsonDict:
    """Enrich a GPU block with torch-derived metrics for *all* GPUs.

    This function is best-effort: if torch/cuda is unavailable, it returns the
    original block unchanged.
    """
    torch = _safe_import("torch")
    if not torch or not isinstance(gpu_block, dict) or not gpu_block.get("gpus"):
        return gpu_block

    try:
        if not torch.cuda.is_available():
            return gpu_block

        device_count = int(torch.cuda.device_count())
        gpus: list[JsonDict] = list(gpu_block.get("gpus") or [])

        # Build an index->entry mapping when possible.
        index_map: dict[int, JsonDict] = {}
        for pos, entry in enumerate(gpus):
            idx = entry.get("index")
            if isinstance(idx, int):
                index_map[idx] = entry
            else:
                index_map[pos] = entry

        for idx in range(device_count):
            try:
                props = torch.cuda.get_device_properties(idx)

                compute_cap = None
                try:
                    cap = torch.cuda.get_device_capability(idx)
                    compute_cap = f"{cap[0]}.{cap[1]}"
                except Exception:
                    compute_cap = None

                sm_count = getattr(props, "multi_processor_count", None)

                l2_bytes = getattr(props, "l2_cache_size", None)
                if not isinstance(l2_bytes, int):
                    l2_bytes = None

                base_clock_mhz = None
                try:
                    # torch reports kHz
                    clock_khz = getattr(props, "clock_rate", None)
                    if clock_khz is not None:
                        base_clock_mhz = float(clock_khz) / 1000.0
                except Exception:
                    base_clock_mhz = None

                cores_per_sm = _cores_per_sm_from_cc(compute_cap)
                total_cores = (
                    int(sm_count) * int(cores_per_sm)
                    if sm_count and cores_per_sm
                    else None
                )
                peak_fp32 = _peak_fp32_gflops(total_cores, base_clock_mhz)

                extra = {
                    "compute_capability": compute_cap,
                    "streaming_multiprocessors": sm_count,
                    "cuda_cores_per_sm": cores_per_sm,
                    "total_cuda_cores": total_cores,
                    "base_clock_mhz": base_clock_mhz,
                    "l2_cache_bytes": int(l2_bytes) if l2_bytes is not None else None,
                    "l2_cache_mb": (
                        float(l2_bytes) / (1024.0**2) if l2_bytes is not None else None
                    ),
                    "peak_fp32_gflops": peak_fp32,
                }

                entry = index_map.get(idx)
                if entry is None:
                    # If mismatch, append a placeholder.
                    entry = {"index": idx}
                    gpus.append(entry)
                    index_map[idx] = entry

                entry.setdefault("derived", {})
                entry["derived"].update({k: v for k, v in extra.items() if v is not None})
            except Exception:
                # Per-device best-effort: continue profiling the remaining GPUs.
                continue

        gpu_block["gpus"] = gpus
        return gpu_block
    except Exception:
        return gpu_block

def enrich_gpu_with_nvml(gpu_block: JsonDict) -> JsonDict:
    """Enrich a GPU block with NVML-derived metrics for *all* GPUs.

    Adds (when available):
    - bus_width_bits
    - memory_clock_mhz
    - graphics_clock_mhz
    - memory_bandwidth_gbs (derived)
    """
    try:
        import pynvml  # type: ignore
    except Exception:
        return gpu_block

    if not isinstance(gpu_block, dict) or not gpu_block.get("gpus"):
        return gpu_block

    try:
        pynvml.nvmlInit()

        gpus: list[JsonDict] = list(gpu_block.get("gpus") or [])

        # Build an index->entry mapping when possible.
        index_map: dict[int, JsonDict] = {}
        for pos, entry in enumerate(gpus):
            idx = entry.get("index")
            if isinstance(idx, int):
                index_map[idx] = entry
            else:
                index_map[pos] = entry

        device_count = int(pynvml.nvmlDeviceGetCount())
        for idx in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

                bus_width = None
                try:
                    bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
                except Exception:
                    bus_width = None

                mem_clock = None
                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM
                    )
                except Exception:
                    mem_clock = None

                gfx_clock = None
                try:
                    gfx_clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                except Exception:
                    gfx_clock = None

                bw = _memory_bandwidth_gbs(bus_width, mem_clock)

                extra = {
                    "bus_width_bits": bus_width,
                    "memory_clock_mhz": mem_clock,
                    "graphics_clock_mhz": gfx_clock,
                    "memory_bandwidth_gbs": bw,
                }

                entry = index_map.get(idx)
                if entry is None:
                    entry = {"index": idx}
                    gpus.append(entry)
                    index_map[idx] = entry

                entry.setdefault("derived", {})
                entry["derived"].update({k: v for k, v in extra.items() if v is not None})
            except Exception:
                continue

        gpu_block["gpus"] = gpus
        pynvml.nvmlShutdown()
        return gpu_block
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return gpu_block

def _populate_ai_and_compat(profile: JsonDict) -> None:
    """Populate AI feature analysis and framework compatibility for detected GPUs.

    This is a best-effort step: it should never raise.
    """
    try:
        gpu_block = profile.get("gpu") or {}
        gpus = list(gpu_block.get("gpus") or [])

        per_gpu: list[JsonDict] = []
        first_cuda_version = None

        # CUDA version may live at different locations depending on collector.
        if isinstance(gpu_block, dict):
            first_cuda_version = (
                gpu_block.get("cuda_version")
                or gpu_block.get("raw", {}).get("cuda_version")
            )

        for i, g in enumerate(gpus):
            derived = g.get("derived") if isinstance(g, dict) else None
            if not isinstance(derived, dict):
                derived = {}

            compute_cap = (
                g.get("compute_cap")
                or derived.get("compute_capability")
                or derived.get("compute_capability")
            )

            if first_cuda_version is None:
                first_cuda_version = g.get("cuda_version")

            mem_bytes = (
                g.get("memory_total_bytes")
                or g.get("total_memory_bytes")
                or g.get("memory_total_bytes")
            )
            mem_gb = (float(mem_bytes) / (1024.0**3)) if mem_bytes else None

            per_gpu.append(
                {
                    "index": g.get("index", i),
                    "name": g.get("name"),
                    "compute_capability": compute_cap,
                    "features": compute_ai_features(compute_cap),
                    "llm_capacity_estimates": estimate_llm_capacity(mem_gb),
                }
            )

        profile["ai"] = {"per_gpu": per_gpu}

        # Compatibility is based on the first GPU.
        first_cap = per_gpu[0].get("compute_capability") if per_gpu else None
        cuda_ver = first_cuda_version or "Unknown"
        profile["compatibility"] = check_software_compatibility(cuda_ver, first_cap)
    except Exception:
        # Keep the profile usable even if AI/compat analysis fails.
        return


def build_profile(include_topology: bool = True) -> JsonDict:
    """Build a full hardware profile.

    The output includes:
    - system: CPU/RAM/Disk/OS info (best-effort)
    - gpu: GPU info (best-effort, NVIDIA-focused)
    - ai: AI feature analysis (based on compute capability) and LLM fit heuristics
    - compatibility: framework compatibility heuristics
    - topology: multi-GPU topology (if enabled and available)
    - collectors: per-collector health checks (attempted/ok/error/duration)
    """
    profile: JsonDict = {
        "system": get_system_info(),
        "gpu": None,
        "topology": None,
        "ai": None,
        "compatibility": None,
        "collectors": {},
    }

    status, smi = _run_collector("nvidia_smi_basic", get_nvidia_smi_basic)
    profile["collectors"][status["name"]] = status

    if smi and isinstance(smi, dict) and smi.get("gpus"):
        profile["gpu"] = smi

        status, enriched = _run_collector(
            "nvml_enrich", enrich_gpu_with_nvml, profile["gpu"]
        )
        profile["collectors"][status["name"]] = status
        profile["gpu"] = enriched or profile["gpu"]

        status, enriched = _run_collector(
            "torch_enrich", enrich_gpu_with_torch, profile["gpu"]
        )
        profile["collectors"][status["name"]] = status
        profile["gpu"] = enriched or profile["gpu"]

        if include_topology:
            status, topo = _run_collector("nvidia_topology", get_nvidia_topology)
            profile["collectors"][status["name"]] = status
            profile["topology"] = topo

        _populate_ai_and_compat(profile)
        return profile

    status, nvml = _run_collector("nvml_basic", get_nvml_info)
    profile["collectors"][status["name"]] = status

    if nvml and isinstance(nvml, dict) and nvml.get("gpus"):
        profile["gpu"] = nvml

        status, enriched = _run_collector(
            "nvml_enrich", enrich_gpu_with_nvml, profile["gpu"]
        )
        profile["collectors"][status["name"]] = status
        profile["gpu"] = enriched or profile["gpu"]

        status, enriched = _run_collector(
            "torch_enrich", enrich_gpu_with_torch, profile["gpu"]
        )
        profile["collectors"][status["name"]] = status
        profile["gpu"] = enriched or profile["gpu"]

        _populate_ai_and_compat(profile)
        return profile

    status, torch_info = _run_collector("torch_basic", get_torch_gpu_info)
    profile["collectors"][status["name"]] = status

    if torch_info:
        profile["gpu"] = torch_info

        status, enriched = _run_collector(
            "torch_enrich", enrich_gpu_with_torch, profile["gpu"]
        )
        profile["collectors"][status["name"]] = status
        profile["gpu"] = enriched or profile["gpu"]

        status, enriched = _run_collector(
            "nvml_enrich", enrich_gpu_with_nvml, profile["gpu"]
        )
        profile["collectors"][status["name"]] = status
        profile["gpu"] = enriched or profile["gpu"]

        _populate_ai_and_compat(profile)
        return profile

    profile["gpu"] = {
        "method": "none",
        "message": "No NVIDIA/CUDA GPU detected (or tools unavailable).",
    }
    return profile

def run_summary(include_topology: bool = True) -> JsonDict:
    """Convenience helper to build a profile and print a human summary."""
    profile = build_profile(include_topology=include_topology)
    print_human_summary(profile)
    return profile


def print_human_summary(profile: JsonDict) -> None:
    """Print a compact, readable summary of the collected profile."""
    sysinfo = profile.get("system", {})
    cpu = sysinfo.get("cpu", {})
    mem = sysinfo.get("memory", {})
    osinfo = sysinfo.get("os", {})

    print("=== Hardware Profiler (agnostic) ===")
    print(f"Timestamp (UTC): {sysinfo.get('timestamp_utc')}")
    print(f"OS: {osinfo.get('system')} {osinfo.get('release')} | Machine: {osinfo.get('machine')}")
    print(
        "Python: "
        f"{sysinfo.get('python', {}).get('version')} | "
        f"Exec: {sysinfo.get('python', {}).get('executable')}"
    )

    if cpu:
        print(
            f"CPU: {cpu.get('physical_cores')} physical cores | "
            f"{cpu.get('logical_cores')} logical cores"
        )
        freq = cpu.get("freq")
        if isinstance(freq, dict):
            print(
                "CPU Freq (MHz): "
                f"current={freq.get('current_mhz')} max={freq.get('max_mhz')}"
            )

    if mem:
        print(
            f"RAM: total={mem.get('ram_total_human')} | "
            f"available={mem.get('ram_available_human')}"
        )
        if mem.get("swap_total_human"):
            print(f"SWAP: total={mem.get('swap_total_human')}")

    gpu_block = profile.get("gpu") or {}
    print("\n--- GPU ---")
    print(f"Detection: {gpu_block.get('method')}")

    gpus = list(gpu_block.get("gpus") or [])
    if not gpus:
        msg = gpu_block.get("message") or gpu_block.get("error")
        if msg:
            print(msg)
    else:
        for g in gpus:
            idx = g.get("index")
            name = g.get("name")
            print(f"GPU[{idx}]: {name}")

            mem_h = g.get("memory_total_human") or g.get("total_memory_human")
            if mem_h:
                print(f"  VRAM total: {mem_h}")

            cap = g.get("compute_cap") or (g.get("derived") or {}).get("compute_capability")
            if cap:
                print(f"  Compute Capability: {cap}")

            derived = g.get("derived") if isinstance(g, dict) else None
            if isinstance(derived, dict) and derived:
                if "memory_bandwidth_gbs" in derived:
                    print(f"  Memory Bandwidth: {derived['memory_bandwidth_gbs']:.2f} GB/s")
                if "l2_cache_mb" in derived:
                    print(f"  L2 Cache: {derived['l2_cache_mb']:.2f} MB")
                if "memory_clock_mhz" in derived:
                    print(f"  Memory Clock: {derived['memory_clock_mhz']} MHz")
                if "bus_width_bits" in derived:
                    print(f"  Bus Width: {derived['bus_width_bits']} bits")

                if "streaming_multiprocessors" in derived:
                    print(f"  Streaming Multiprocessors: {derived['streaming_multiprocessors']}")
                if "cuda_cores_per_sm" in derived:
                    print(f"  CUDA Cores per SM: {derived['cuda_cores_per_sm']}")
                if "total_cuda_cores" in derived:
                    total = int(derived["total_cuda_cores"])
                    print(f"  Total CUDA Cores: {total:,}".replace(",", "."))
                if "base_clock_mhz" in derived:
                    print(f"  Base Clock: {derived['base_clock_mhz']:.0f} MHz")
                if "peak_fp32_gflops" in derived:
                    print(f"  Peak FP32 Performance: {derived['peak_fp32_gflops']:.0f} GFLOPS")

    ai = profile.get("ai") or {}
    per_gpu = ai.get("per_gpu") if isinstance(ai, dict) else None
    if isinstance(per_gpu, list) and per_gpu:
        print("\n--- AI Features ---")
        for item in per_gpu:
            idx = item.get("index")
            feats = (item.get("features") or {}) if isinstance(item, dict) else {}
            if feats:
                print(f"GPU[{idx}] AI Hardware Rating: {feats.get('ai_hardware_rating')}")
                print(f"  Tensor Cores: {feats.get('tensor_cores')}")
                prec = feats.get("supported_precisions") or []
                print(f"  Supported Precisions: {', '.join(prec)}")

    comp = profile.get("compatibility")
    if isinstance(comp, dict) and comp.get("frameworks"):
        print("\n--- Compatibility (heuristic) ---")
        for fw, st in comp["frameworks"].items():
            ok = "OK" if st.get("supported") else "NOK"
            print(
                f"{fw}: {ok} (min CUDA {st.get('min_cuda')}, "
                f"min CC {st.get('min_compute')})"
            )

    collectors = profile.get("collectors")
    if isinstance(collectors, dict) and collectors:
        print("\n--- Collectors health ---")
        for name, st in collectors.items():
            if not isinstance(st, dict):
                continue
            status = "OK" if st.get("ok") else "FAIL"
            dur = st.get("duration_ms")
            err = st.get("error")
            if err:
                print(f"{name}: {status} ({dur} ms) - {err}")
            else:
                print(f"{name}: {status} ({dur} ms)")

    topo = profile.get("topology")
    if isinstance(topo, dict) and topo.get("matrix"):
        print("\n--- Topology (nvidia-smi topo -m) ---")
        print(topo["matrix"])

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Portable hardware profiler.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full profile as JSON.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write JSON output to a file (e.g., profile.json).",
    )
    parser.add_argument(
        "--no-topology",
        action="store_true",
        help="Skip multi-GPU/PCIe topology probing.",
    )
    args = parser.parse_args()

    profile = build_profile(include_topology=not args.no_topology)

    if args.json or args.out:
        payload = json.dumps(
            profile,
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as file:
                file.write(payload)
        if args.json:
            print(payload)
    else:
        print_human_summary(profile)


if __name__ == "__main__":
    main()

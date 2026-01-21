"""GPU monitoring utilities (best-effort, NVML-based).

This module provides lightweight sampling of GPU telemetry over time.
The dependency is optional; install with:

    pip install -e ".[nvml]"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import time


class NvmlUnavailable(RuntimeError):
    """Raised when NVML (pynvml) is not available."""


def _try_import_nvml():
    try:
        import pynvml  # type: ignore
        return pynvml
    except Exception:
        return None


@dataclass
class MonitorConfig:
    """Configuration for GPU telemetry sampling."""

    interval_s: float = 0.5
    duration_s: float = 10.0
    include_power: bool = True
    include_temperature: bool = True
    include_utilization: bool = True
    include_memory: bool = True


def sample_gpu_telemetry(
    *,
    config: Optional[MonitorConfig] = None,
    gpu_indices: Optional[List[int]] = None,
    best_effort: bool = True,
) -> Dict[str, Any]:
    """Sample GPU telemetry over time using NVML."""
    cfg = config or MonitorConfig()
    pynvml = _try_import_nvml()
    if pynvml is None:
        if best_effort:
            return {
                "ok": False,
                "error": "pynvml (NVML) is not installed or not supported in this environment.",
                "samples": [],
            }
        raise NvmlUnavailable("pynvml (NVML) is required for GPU monitoring.")

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        indices = gpu_indices if gpu_indices is not None else list(range(count))
        handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices}
    except Exception as exc:
        if best_effort:
            return {"ok": False, "error": str(exc), "samples": []}
        raise

    samples: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    next_t = t0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - t0
            if elapsed >= cfg.duration_s:
                break

            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            next_t = now + cfg.interval_s

            entry: Dict[str, Any] = {"t_s": round(elapsed, 6), "gpus": {}}
            for idx, h in handles.items():
                g: Dict[str, Any] = {}

                if cfg.include_memory:
                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        g["mem_total_bytes"] = int(mem.total)
                        g["mem_used_bytes"] = int(mem.used)
                        g["mem_free_bytes"] = int(mem.free)
                    except Exception:
                        pass

                if cfg.include_utilization:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(h)
                        g["util_gpu_pct"] = int(util.gpu)
                        g["util_mem_pct"] = int(util.memory)
                    except Exception:
                        pass

                if cfg.include_power:
                    try:
                        g["power_w"] = float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
                    except Exception:
                        pass

                if cfg.include_temperature:
                    try:
                        g["temp_c"] = int(
                            pynvml.nvmlDeviceGetTemperature(
                                h, pynvml.NVML_TEMPERATURE_GPU
                            )
                        )
                    except Exception:
                        pass

                entry["gpus"][str(idx)] = g

            samples.append(entry)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return {
        "ok": True,
        "config": {
            "interval_s": cfg.interval_s,
            "duration_s": cfg.duration_s,
            "include_power": cfg.include_power,
            "include_temperature": cfg.include_temperature,
            "include_utilization": cfg.include_utilization,
            "include_memory": cfg.include_memory,
        },
        "samples": samples,
    }

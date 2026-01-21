"""Runtime profiling utilities for PyTorch workloads.

This module complements :mod:`hw_profiler.hardware` by providing execution-time
profiling. It helps answer: *where is time spent while running the model?*

It wraps ``torch.profiler`` with a small, stable API. The dependency is optional:
install it with:

    pip install -e ".[torch]"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import time


class TorchProfilerUnavailable(RuntimeError):
    """Raised when torch.profiler is not available in the environment."""


def _require_torch_profiler() -> None:
    try:
        import torch  # noqa: F401
        from torch import profiler as _  # noqa: F401
    except Exception as exc:
        raise TorchProfilerUnavailable(
            "PyTorch profiler is unavailable. Install torch and use a supported build."
        ) from exc


@dataclass
class ProfileConfig:
    """Configuration for torch.profiler schedule and options."""

    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False


def profile_model_performance(
    step_fn: Callable[[], Any],
    *,
    num_steps: int = 10,
    use_cuda: Optional[bool] = None,
    config: Optional[ProfileConfig] = None,
) -> Dict[str, Any]:
    """Profile a representative workload step function with torch.profiler."""
    _require_torch_profiler()
    import torch
    from torch import profiler as torch_profiler

    cfg = config or ProfileConfig()
    if use_cuda is None:
        use_cuda = bool(torch.cuda.is_available())

    activities = [torch_profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch_profiler.ProfilerActivity.CUDA)

    schedule = torch_profiler.schedule(
        wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=cfg.repeat
    )

    meta: Dict[str, Any] = {
        "num_steps": num_steps,
        "use_cuda": use_cuda,
        "schedule": {
            "wait": cfg.wait,
            "warmup": cfg.warmup,
            "active": cfg.active,
            "repeat": cfg.repeat,
        },
        "options": {
            "record_shapes": cfg.record_shapes,
            "profile_memory": cfg.profile_memory,
            "with_stack": cfg.with_stack,
            "with_flops": cfg.with_flops,
        },
    }

    start = time.perf_counter()
    with torch_profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
    ) as prof:
        for _ in range(num_steps):
            step_fn()
            prof.step()

    duration_ms = (time.perf_counter() - start) * 1000.0
    table = prof.key_averages().table(
        sort_by="cuda_time_total" if use_cuda else "cpu_time_total",
        row_limit=50,
    )

    return {
        "events_table": table,
        "metadata": meta,
        "collector": {"ok": True, "duration_ms": duration_ms},
    }


def summarize_top_ops(prof_result: Dict[str, Any], *, top_k: int = 20) -> Dict[str, Any]:
    """Return a lightweight summary of top ops from a profiler result."""
    table = prof_result.get("events_table", "")
    lines = [ln.rstrip() for ln in str(table).splitlines() if ln.strip()]

    rows: List[str] = []
    started = False
    for ln in lines:
        if set(ln.strip()) <= {"-", " "} and len(ln.strip()) >= 3:
            started = True
            continue
        if started:
            rows.append(ln)

    return {
        "top_k": top_k,
        "rows": rows[:top_k],
        "note": "Rows are kept as strings for robustness across torch versions.",
    }


def export_chrome_trace(
    step_fn: Callable[[], Any],
    *,
    num_steps: int = 10,
    out_path: str | Path = "trace.json",
    use_cuda: Optional[bool] = None,
    config: Optional[ProfileConfig] = None,
) -> str:
    """Export a Chrome trace JSON (timeline) from torch.profiler."""
    _require_torch_profiler()
    import torch
    from torch import profiler as torch_profiler

    cfg = config or ProfileConfig()
    if use_cuda is None:
        use_cuda = bool(torch.cuda.is_available())

    activities = [torch_profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch_profiler.ProfilerActivity.CUDA)

    schedule = torch_profiler.schedule(
        wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=cfg.repeat
    )

    out_path = str(Path(out_path))
    with torch_profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
    ) as prof:
        for _ in range(num_steps):
            step_fn()
            prof.step()

    try:
        prof.export_chrome_trace(out_path)
    except Exception:
        pass

    return out_path

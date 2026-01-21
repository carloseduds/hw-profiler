"""hw_profiler package.

Portable profiling utilities designed to run from PCs to edge devices.

Public API:
- Hardware profiling: build_profile, print_human_summary, run_summary

Optional modules:
- Runtime profiling (requires torch): hw_profiler.torch_profiler
- GPU monitoring (requires pynvml): hw_profiler.gpu_monitor
"""

from .hardware import build_profile, print_human_summary, run_summary

__all__ = ["build_profile", "print_human_summary", "run_summary"]

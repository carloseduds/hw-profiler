"""Hardware profiling collectors and reporting.

This module contains functions that collect *static* hardware capabilities and
basic environment details, designed to work from laptops/PCs to edge devices.

Notes
-----
- The implementation lives in :mod:`hw_profiler.core` for backward compatibility.
- This file provides a stable import path and a clearer public API.
"""

from __future__ import annotations

from .core import build_profile, print_human_summary, run_summary

__all__ = ["build_profile", "print_human_summary", "run_summary"]

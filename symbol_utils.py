#!/usr/bin/env python3
"""
symbol_utils.py — C++/CUDA symbol demangling and kernel name analysis.
"""

import re
import subprocess
from functools import lru_cache


@lru_cache(maxsize=4096)
def demangle_symbol(mangled: str) -> str:
    """Demangle a C++/CUDA symbol using c++filt."""
    try:
        result = subprocess.run(
            ["c++filt", "-n", mangled],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return mangled


def batch_demangle(symbols: list) -> dict:
    """Demangle multiple symbols efficiently."""
    if not symbols:
        return {}

    try:
        proc = subprocess.run(
            ["c++filt", "-n"] + list(symbols),
            capture_output=True, text=True, timeout=30,
        )
        lines = proc.stdout.strip().split("\n")
        return {m: d for m, d in zip(symbols, lines)}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {s: s for s in symbols}


def classify_kernel(name: str) -> str:
    """Classify a CUDA kernel by its name pattern."""
    demangled = demangle_symbol(name)

    # NCCL kernels
    if "nccl" in demangled.lower():
        for proto in ["LL128", "LL", "Simple"]:
            if proto in demangled:
                return f"NCCL/{proto}"
        if "AllReduce" in demangled:
            return "NCCL/AllReduce"
        if "AllGather" in demangled:
            return "NCCL/AllGather"
        if "ReduceScatter" in demangled:
            return "NCCL/ReduceScatter"
        if "Broadcast" in demangled:
            return "NCCL/Broadcast"
        return "NCCL"

    # cuBLAS kernels
    if "cublas" in demangled.lower() or "gemm" in demangled.lower():
        return "cuBLAS"

    # cuDNN kernels
    if "cudnn" in demangled.lower():
        return "cuDNN"

    # PyTorch native kernels
    if "at::" in demangled or "aten::" in demangled:
        return "ATen"

    # Elementwise / copy
    if "vectorized_elementwise" in demangled:
        return "Elementwise"
    if "memcpy" in demangled or "memset" in demangled:
        return "MemOps"

    return "Other"


def is_nccl_kernel(name: str) -> bool:
    """Check if a kernel name belongs to NCCL."""
    return "nccl" in name.lower() or "ncclDev" in name


def extract_kernel_metadata(ptx_text: str) -> list:
    """Extract kernel names and metadata from PTX text."""
    kernels = []
    current = None

    for line in ptx_text.split("\n"):
        # Entry point (kernel function)
        m = re.match(r"\.visible\s+\.entry\s+(\S+)\(", line)
        if not m:
            m = re.match(r"\.entry\s+(\S+)\(", line)
        if m:
            if current:
                kernels.append(current)
            name = m.group(1)
            current = {
                "mangled_name": name,
                "demangled_name": demangle_symbol(name),
                "classification": classify_kernel(name),
                "params": [],
                "registers": 0,
                "shared_mem": 0,
                "lines": 0,
                "start_line": 0,
            }
            continue

        if current:
            current["lines"] += 1

            # Parameters
            pm = re.match(r"\.param\s+\.\w+\s+(\S+)", line)
            if pm:
                current["params"].append(pm.group(1))

            # Register count
            rm = re.match(r"\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>", line)
            if rm:
                count = int(rm.group(3))
                current["registers"] += count

    if current:
        kernels.append(current)

    return kernels


def format_kernel_header(meta: dict) -> str:
    """Format a kernel metadata header."""
    lines = [
        f"{'=' * 72}",
        f"  Kernel: {meta['demangled_name']}",
        f"  Mangled: {meta['mangled_name']}",
        f"  Category: {meta['classification']}",
        f"  Parameters: {len(meta.get('params', []))}",
        f"  Registers (approx): {meta.get('registers', '?')}",
        f"  PTX Lines: {meta.get('lines', '?')}",
        f"{'=' * 72}",
    ]
    return "\n".join(lines)

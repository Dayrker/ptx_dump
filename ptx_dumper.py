#!/usr/bin/env python3
"""
ptx_dumper.py — Orchestrate PTX/SASS extraction from multiple sources.

Strategy (in priority order):
1. Pre-compiled PTX from .so files (cuobjdump -ptx) — requires compute_XX build
2. JIT-compiled PTX from CUDA_JIT_CACHE_DIR
3. Existing PTX dump files (from prior compilations)
4. SASS extraction (cuobjdump -sass) — always available as fallback

SASS is the actual GPU machine code and is always available.
PTX is the virtual assembly and requires specific build flags.
"""

import os
import re
import glob
import subprocess
from typing import Optional

from env_setup import EnvConfig
from symbol_utils import is_nccl_kernel, demangle_symbol
from ptx_formatter import format_ptx_section, write_formatted_ptx


# ─── SASS Formatter ──────────────────────────────────────────────────


def _parse_sass_into_kernels(sass_text: str) -> list:
    """Parse SASS text into per-kernel blocks."""
    kernels = []
    current_name = None
    current_lines = []

    for line in sass_text.split("\n"):
        # Function header
        m = re.match(r"\s+Function\s*:\s*(\S+)", line)
        if m:
            if current_name:
                kernels.append({
                    "name": current_name,
                    "demangled": demangle_symbol(current_name),
                    "lines": current_lines,
                })
            current_name = m.group(1)
            current_lines = [line]
        elif current_name:
            current_lines.append(line)

    if current_name:
        kernels.append({
            "name": current_name,
            "demangled": demangle_symbol(current_name),
            "lines": current_lines,
        })

    return kernels


def _format_sass_kernel(kernel: dict) -> str:
    """Format a single SASS kernel with header."""
    lines = [
        "=" * 72,
        f"  Kernel: {kernel['demangled']}",
        f"  Mangled: {kernel['name']}",
        f"  SASS Lines: {len(kernel['lines'])}",
        "=" * 72,
        "",
    ]
    lines.extend(kernel["lines"])
    return "\n".join(lines)


def write_formatted_sass(sass_text: str, output_dir: str, prefix: str = "dump",
                         arch: str = "sm_80",
                         nccl_only: bool = False) -> list:
    """Write formatted SASS to files, one per kernel."""
    os.makedirs(output_dir, exist_ok=True)
    written = []

    kernels = _parse_sass_into_kernels(sass_text)
    if not kernels:
        return written

    # Filter NCCL if requested
    if nccl_only:
        kernels = [k for k in kernels if is_nccl_kernel(k["name"])]

    # Summary
    summary_lines = [
        "=" * 72,
        f"  SASS Dump Summary ({arch})",
        f"  NCCL-only: {nccl_only}",
        f"  Total kernels: {len(kernels)}",
        "=" * 72,
        "",
    ]

    from symbol_utils import classify_kernel
    categories = {}
    for k in kernels:
        cat = classify_kernel(k["name"])
        categories.setdefault(cat, []).append(k)

    for cat in sorted(categories.keys()):
        ks = categories[cat]
        summary_lines.append(f"  [{cat}] — {len(ks)} kernel(s)")
        summary_lines.append(f"  {'─' * 50}")
        for k in ks[:50]:  # cap display
            summary_lines.append(f"    • {k['demangled']}  ({len(k['lines'])} lines)")
        if len(ks) > 50:
            summary_lines.append(f"    ... +{len(ks) - 50} more")
        summary_lines.append("")

    summary_path = os.path.join(output_dir, "SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    written.append(summary_path)

    # Combined file
    combined_path = os.path.join(output_dir, f"{prefix}_all_sass.txt")
    with open(combined_path, "w") as f:
        f.write("\n\n".join(_format_sass_kernel(k) for k in kernels))
    written.append(combined_path)

    # Per-kernel files
    for i, k in enumerate(kernels):
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", k["demangled"])[:80]
        filename = f"{prefix}_sass_{i + 1:03d}_{safe_name}.sass"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(_format_sass_kernel(k))
        written.append(filepath)

    return written


# ─── PTX Dumper ──────────────────────────────────────────────────────


class PTXDumper:
    """Extracts and formats PTX/SASS from various sources."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.cuobjdump = config.cuobjdump
        self.collected_ptx = {}   # source -> ptx_text
        self.collected_sass = {}  # source -> sass_text
        self.kernel_sources = {}  # kernel_name -> source

    # ── PTX extraction ──

    def dump_ptx_from_lib(self, lib_path: str, label: str = "lib") -> str:
        """Extract PTX from a .so/.cubin using cuobjdump -ptx."""
        try:
            result = subprocess.run(
                [self.cuobjdump, "-ptx", lib_path],
                capture_output=True, text=True, timeout=120,
            )
            ptx = result.stdout
            if ptx and ".version" in ptx:
                self.collected_ptx[f"{label}_ptx"] = ptx
                self._tag_kernels_ptx(ptx, f"ptx:{os.path.basename(lib_path)}")
                return ptx
        except subprocess.TimeoutExpired:
            pass
        return ""

    def dump_ptx_from_file(self, file_path: str, label: str = "file") -> str:
        """Read an existing PTX dump file."""
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                ptx = f.read()
            if ptx and (".version" in ptx or ".entry" in ptx):
                self.collected_ptx[f"{label}_ptx"] = ptx
                self._tag_kernels_ptx(ptx, f"ptx_file:{os.path.basename(file_path)}")
                return ptx
        return ""

    def dump_ptx_from_jit_cache(self, cache_dir: str) -> str:
        """Extract PTX from JIT cache directory."""
        all_ptx = []
        if not os.path.isdir(cache_dir):
            return ""
        for fpath in glob.glob(os.path.join(cache_dir, "**", "*"), recursive=True):
            if os.path.isfile(fpath):
                ptx = self.dump_ptx_from_lib(fpath, f"jit_{os.path.basename(fpath)}")
                if ptx:
                    all_ptx.append(ptx)
        combined = "\n\n".join(all_ptx)
        if combined:
            self.collected_ptx["jit_ptx"] = combined
        return combined

    # ── SASS extraction ──

    def dump_sass_from_lib(self, lib_path: str, arch: str = "sm_80",
                           label: str = "lib") -> str:
        """Extract SASS from a .so using cuobjdump -sass."""
        try:
            result = subprocess.run(
                [self.cuobjdump, "-sass", "-arch", arch, lib_path],
                capture_output=True, text=True, timeout=600,
            )
            sass = result.stdout
            if sass and "Function" in sass:
                self.collected_sass[f"{label}_sass"] = sass
                self._tag_kernels_sass(sass, f"sass:{os.path.basename(lib_path)}")
                return sass
        except subprocess.TimeoutExpired:
            print(f"  [WARN] SASS extraction timed out for {lib_path}")
        return ""

    # ── NCCL-specific ──

    def dump_nccl_ptx(self) -> str:
        """Try to extract PTX from local NCCL lib."""
        return self.dump_ptx_from_lib(self.config.nccl_lib, "nccl")

    def dump_nccl_sass(self, arch: str = "sm_80") -> str:
        """Extract SASS from local NCCL lib."""
        return self.dump_sass_from_lib(self.config.nccl_lib, arch=arch, label="nccl")

    def dump_existing_nccl_ptx(self) -> str:
        """Use an existing PTX dump file if available."""
        # Check common locations
        candidates = [
            os.path.join(self.config.nccl_ptx_dir, "libnccl.so.2.21.5.compiled.sm_80.ptx"),
            os.path.join(self.config.project_dir, "nccl_ptx", "libnccl.so.2.21.5.compiled.sm_80.ptx"),
            os.path.join(self.config.project_dir, "nccl_ptx", "nccl_ptx_from_cuobjdump.txt"),
        ]
        for path in candidates:
            ptx = self.dump_ptx_from_file(path, "existing_nccl")
            if ptx:
                print(f"  [OK] Using existing PTX dump: {os.path.basename(path)}")
                return ptx
        return ""

    # ── Filtering ──

    def filter_nccl_ptx(self, ptx_text: str) -> str:
        """Filter PTX to only NCCL kernels."""
        if not ptx_text:
            return ""
        lines = ptx_text.split("\n")
        output = []
        in_nccl = False
        brace_depth = 0

        for line in lines:
            m = re.match(r"\.?\s*\.?(visible\s+)?\.entry\s+(\S+)", line)
            if m:
                name = m.group(2) if m.group(2) else ""
                in_nccl = is_nccl_kernel(name)
                if in_nccl:
                    brace_depth = 0

            if in_nccl:
                output.append(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0 and "}" in line and "{" in "".join(output[-20:]):
                    in_nccl = False
                    output.append("")

        # Keep header
        header = []
        for line in lines:
            if re.match(r"\.?\s*\.?(visible\s+)?\.entry\s+", line):
                break
            header.append(line)

        return "\n".join(header) + "\n\n" + "\n".join(output) if output else ""

    def filter_nccl_sass(self, sass_text: str) -> str:
        """Filter SASS to only NCCL kernels."""
        if not sass_text:
            return ""
        kernels = _parse_sass_into_kernels(sass_text)
        nccl_kernels = [k for k in kernels if is_nccl_kernel(k["name"])]
        return "\n\n".join(_format_sass_kernel(k) for k in nccl_kernels)

    # ── Internal ──

    def _tag_kernels_ptx(self, ptx_text: str, source: str):
        for line in ptx_text.split("\n"):
            m = re.match(r"\.?\s*\.?(visible\s+)?\.entry\s+(\S+)", line)
            if m:
                name = m.group(2) if m.group(2) else m.group(1)
                if name:
                    self.kernel_sources[name] = source

    def _tag_kernels_sass(self, sass_text: str, source: str):
        for line in sass_text.split("\n"):
            m = re.match(r"\s+Function\s*:\s*(\S+)", line)
            if m:
                self.kernel_sources[m.group(1)] = source

    # ── Output ──

    def write_output(self, output_dir: str, nccl_only: bool = False,
                     prefix: str = "dump", arch: str = "sm_80") -> list:
        """Write all collected PTX/SASS to formatted output files."""
        os.makedirs(output_dir, exist_ok=True)
        written = []

        # 1. Write PTX if available
        all_ptx = "\n\n".join(self.collected_ptx.values())
        if all_ptx.strip():
            if nccl_only:
                all_ptx = self.filter_nccl_ptx(all_ptx)
            if all_ptx.strip():
                files = write_formatted_ptx(all_ptx, output_dir, prefix=prefix,
                                           split_per_kernel=True)
                written.extend(files)
                print(f"  [OK] PTX written ({len(files)} files)")

        # 2. Write SASS if available (always the fallback)
        all_sass = "\n\n".join(self.collected_sass.values())
        if all_sass.strip():
            sass_files = write_formatted_sass(
                all_sass, output_dir, prefix=prefix,
                arch=arch, nccl_only=nccl_only,
            )
            written.extend(sass_files)
            print(f"  [OK] SASS written ({len(sass_files)} files)")

        if not written:
            print(f"  [WARN] No PTX or SASS collected.")

        return written


# ─── High-level functions ─────────────────────────────────────────────


def dump_single_gpu_ptx(config: EnvConfig, trace_calls: bool = False) -> list:
    """
    Dump all PTX/SASS for single-GPU mode.

    Captures: JIT PTX → NCCL lib PTX → NCCL SASS (fallback).
    """
    dumper = PTXDumper(config)
    found = False

    # 1. JIT cache
    jit_dir = os.path.join(config.project_dir, ".jit_cache")
    jit_ptx = dumper.dump_ptx_from_jit_cache(jit_dir)
    if jit_ptx:
        print(f"  [OK] JIT PTX extracted ({len(jit_ptx)} chars)")
        found = True
    else:
        print(f"  [INFO] No JIT PTX (kernels are pre-compiled)")

    # 2. NCCL lib PTX
    nccl_ptx = dumper.dump_nccl_ptx()
    if nccl_ptx:
        print(f"  [OK] NCCL PTX extracted ({len(nccl_ptx)} chars)")
        found = True

    # 3. Existing PTX dump
    if not found:
        existing = dumper.dump_existing_nccl_ptx()
        if existing:
            found = True

    # 4. SASS fallback (always available)
    print(f"  [INFO] Extracting SASS from NCCL lib (sm_80)...")
    sass = dumper.dump_nccl_sass()
    if sass:
        print(f"  [OK] NCCL SASS extracted ({len(sass)} chars, "
              f"{sass.count('Function')} functions)")
        found = True

    # Write output
    files = dumper.write_output(config.single_ptx_dir, nccl_only=False, prefix="single")
    print(f"  [OK] Wrote {len(files)} files to {config.single_ptx_dir}")
    return files


def dump_dual_gpu_ptx(config: EnvConfig, nccl_only: bool = False,
                      trace_calls: bool = False) -> list:
    """
    Dump PTX/SASS for dual-GPU mode (NCCL-focused).

    Captures: JIT PTX → NCCL lib PTX → existing PTX → SASS.
    """
    dumper = PTXDumper(config)
    found = False

    # 1. JIT cache
    jit_dir = os.path.join(config.project_dir, ".jit_cache")
    jit_ptx = dumper.dump_ptx_from_jit_cache(jit_dir)
    if jit_ptx:
        print(f"  [OK] JIT PTX extracted ({len(jit_ptx)} chars)")
        found = True

    # 2. NCCL lib PTX
    nccl_ptx = dumper.dump_nccl_ptx()
    if nccl_ptx:
        print(f"  [OK] NCCL PTX from lib ({len(nccl_ptx)} chars)")
        found = True

    # 3. Existing PTX dump
    if not found:
        existing = dumper.dump_existing_nccl_ptx()
        if existing:
            found = True

    # 4. SASS fallback
    print(f"  [INFO] Extracting SASS from NCCL lib (sm_80)...")
    sass = dumper.dump_nccl_sass()
    if sass:
        n_funcs = sass.count("Function")
        print(f"  [OK] NCCL SASS extracted ({len(sass)} chars, {n_funcs} functions)")
        found = True

    # Write output
    files = dumper.write_output(config.nccl_ptx_dir, nccl_only=nccl_only, prefix="nccl")
    print(f"  [OK] Wrote {len(files)} files to {config.nccl_ptx_dir}")
    return files

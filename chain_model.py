#!/usr/bin/env python3
"""
chain_model.py — Data model for the per-kernel call chain.

A CallChain links a single CUDA kernel (the PTX we dump) back up through the
layers that reached it:

    python  (torch 算子)   ── model.generate / F.linear / dist.all_reduce
    aten    (ATen op)      ── aten::mm / c10d::allreduce_
    runtime (底层 runtime)  ── ProcessGroupNCCL → ncclAllReduce → ncclLaunchKernel …
    cuda_launch             ── cudaLaunchKernel (correlation id)
    kernel                  ── the device kernel name (profiler form)
    ptx                     ── the dumped .ptx file for this kernel

The "runtime" layers between aten and the device kernel are *not* directly
observable from the profiler — they are static knowledge encoded in
runtime_knowledge.py (sourced from docs/allreduce-deep-dive.md). The python /
aten / kernel / launch layers ARE observed (from the chrome trace, all on one
clock and joined by args.correlation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# Layer tags (Frame.layer values). Ordered top → bottom.
LAYER_PYTHON = "python"
LAYER_ATEN = "aten"
LAYER_RUNTIME = "runtime"
LAYER_LAUNCH = "launch"
LAYER_KERNEL = "kernel"
LAYER_PTX = "ptx"

LAYER_ORDER = {
    LAYER_PYTHON: 0, LAYER_ATEN: 1, LAYER_RUNTIME: 2,
    LAYER_LAUNCH: 3, LAYER_KERNEL: 4, LAYER_PTX: 5,
}


@dataclass
class Frame:
    """One hop in the call chain."""
    layer: str          # one of the LAYER_* constants
    name: str           # human-readable label (demangled / function signature)
    detail: Dict[str, Any] = field(default_factory=dict)
    source: str = "observed"   # "observed" (profiler) | "static" (knowledge table)

    def render(self) -> str:
        """One-line rendering for the text report / PTX header."""
        tag = f"[{self.layer:7s}]"
        src = "" if self.source == "observed" else f"  ({self.source})"
        bits = [self.name]
        # Append compact detail (skip empty / huge fields)
        for k in ("shapes", "seq", "grid", "block", "smem", "corr", "count",
                  "cuda_time", "note", "file", "ptx_file"):
            v = self.detail.get(k)
            if v not in (None, "", [], ()):
                bits.append(f"{k}={v}")
        return f"  {tag} {', '.join(bits)}{src}"


@dataclass
class Caller:
    """A representative (aten op, shapes) pair that reaches a shared kernel."""
    aten_op: str
    shapes: Any = None

    def render(self) -> str:
        return f"  {self.aten_op}  shapes={self.shapes}" if self.shapes else f"  {self.aten_op}"


@dataclass
class CallChain:
    """Full chain for one unique kernel that ran during the traced region."""
    kernel_profiler_name: str           # name as profiler reports it
    kernel_mangled_name: Optional[str] = None   # matched PTX .entry symbol (None if unmatched)
    ptx_file: Optional[str] = None       # relative path of the dumped .ptx file
    category: str = "Other"             # from symbol_utils.classify_kernel
    frames: List[Frame] = field(default_factory=list)   # ordered top → bottom
    occurrence_count: int = 1
    total_cuda_time_us: float = 0.0
    representative_callers: List[Caller] = field(default_factory=list)
    matched: bool = False
    match_note: str = ""

    def add_runtime_layers(self, layers: List[Frame]):
        """Insert static runtime frames just before the launch/kernel frames."""
        insert_at = len(self.frames)
        for i, f in enumerate(self.frames):
            if f.layer in (LAYER_LAUNCH, LAYER_KERNEL, LAYER_PTX):
                insert_at = i
                break
        self.frames[insert_at:insert_at] = layers

    def render_block(self) -> str:
        """Render the whole chain as a text block (for CALL_CHAINS.txt)."""
        lines = []
        lines.append("=" * 78)
        lines.append(f"  Kernel: {self.kernel_profiler_name}")
        if self.kernel_mangled_name and self.kernel_mangled_name != self.kernel_profiler_name:
            lines.append(f"  Mangled: {self.kernel_mangled_name}")
        lines.append(f"  Category: {self.category}  |  calls: {self.occurrence_count}"
                     f"  |  total CUDA: {self.total_cuda_time_us / 1000:.2f} ms")
        if self.matched:
            lines.append(f"  PTX: {self.ptx_file or '(in combined file)'}")
        else:
            lines.append(f"  PTX: (no PTX block matched — {self.match_note or 'kernel not in dumped libs'})")
        lines.append("-" * 78)
        lines.append("  Call chain (torch op → ATen → runtime → kernel → PTX):")
        for f in self.frames:
            lines.append(f.render())
        if self.representative_callers and self.occurrence_count > 1:
            lines.append("-" * 78)
            lines.append(f"  Representative callers ({len(self.representative_callers)} shown"
                         f" of {self.occurrence_count} launches):")
            for c in self.representative_callers:
                lines.append(c.render())
        lines.append("=" * 78)
        return "\n".join(lines)


def format_chain_header(chain: Optional[CallChain]) -> str:
    """Compact chain block to prepend to a per-kernel .ptx file.

    Returns "" if no chain is attached (keeps PTX files clean when unmatched).
    """
    if chain is None:
        return ""
    lines = [
        "─" * 72,
        "  Call Chain (torch op → ATen → runtime → kernel → PTX)",
        "─" * 72,
    ]
    for f in chain.frames:
        lines.append(f.render())
    lines.append("─" * 72)
    lines.append("")
    return "\n".join(lines)


def chain_to_jsonable(chain: CallChain) -> Dict[str, Any]:
    """JSON-serializable form for call_chains.json."""
    return {
        "kernel_profiler_name": chain.kernel_profiler_name,
        "kernel_mangled_name": chain.kernel_mangled_name,
        "ptx_file": chain.ptx_file,
        "category": chain.category,
        "occurrence_count": chain.occurrence_count,
        "total_cuda_time_us": chain.total_cuda_time_us,
        "matched": chain.matched,
        "match_note": chain.match_note,
        "frames": [
            {"layer": f.layer, "name": f.name, "detail": f.detail, "source": f.source}
            for f in chain.frames
        ],
        "representative_callers": [asdict(c) for c in chain.representative_callers],
    }

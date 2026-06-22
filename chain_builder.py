#!/usr/bin/env python3
"""
chain_builder.py — Reconstruct per-kernel call chains from a torch.profiler
chrome trace.

The profiler records four event categories on ONE absolute-microsecond clock:

    python_function  ── the Python torch op (model.generate, F.linear, …)
    cpu_op           ── the ATen op (aten::mm, c10d::allreduce_, …)
    cuda_runtime     ── cudaLaunchKernel / cudaLaunchKernelExC (carries correlation)
    kernel           ── the device kernel (carries the same correlation)

The chain closes by correlation + time-containment:

    kernel.args.correlation  ──match──▶  cuda_runtime (same correlation)
    cuda_runtime.ts           ─contains──▶ one or more cpu_op[ts, ts+dur]
    innermost cpu_op.ts       ─contains──▶ the enclosing python_function frames

The "runtime" layers between ATen and the kernel (ProcessGroupNCCL →
ncclAllReduce → …, or cublasGemmEx) are static knowledge from
runtime_knowledge.py — not observable, but deterministic per kernel family.

Note: torch 2.5.1's FunctionEvent.stack is always empty even with
with_stack=True; the chrome trace's python_function events are the reliable
source of the Python layer (verified).
"""

from __future__ import annotations

import os
import json
import tempfile
from collections import defaultdict
from typing import List

from chain_model import (CallChain, Frame, Caller,
                         LAYER_PYTHON, LAYER_ATEN, LAYER_LAUNCH, LAYER_KERNEL)
from symbol_utils import is_nccl_kernel, classify_kernel
import runtime_knowledge


# ─── Interval index (time containment) ─────────────────────────────────


class _IntervalIndex:
    """Sorted-by-start list of {ts, dur, ...} events; query which contain a ts."""

    def __init__(self, events: list):
        self.events = sorted(
            [e for e in events if e.get("ts") is not None and e.get("dur", 0) >= 0],
            key=lambda e: e["ts"],
        )

    def containing(self, ts: float) -> list:
        """All events whose [ts, ts+dur] covers `ts`, outer → inner."""
        out = []
        for e in self.events:
            if e["ts"] > ts:
                break
            if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
                out.append(e)
        # outer (earliest start) first → inner (latest start) last
        out.sort(key=lambda e: e["ts"])
        return out


# ─── Python frame filtering ───────────────────────────────────────────

# Frames whose name matches these are pure plumbing — drop them.
_PLUMBING = (
    "<built-in", "type object at", "<method", " __getattr__", "__setattr__",
    "isinstance(", "hasattr(", "getattr(", "<frozen", "site-packages/torch",
    "lib/python3", "/python3.", "/logging/", "/os.py", "/warnings.py",
    "/inspect.py", "/functools.py", "/typing.py", "/abc.py",
    # threads / progress bars / torch internals that aren't the user's call path
    "threading.py", "tqdm/", "_contextlib.py", "contextlib.py",
    "generic.py", "deprecation.py", "module.py(1740", "_call_impl",
    "nn.Module:", "decorate_context", "device_sync",
)


def _is_plumbing(name: str) -> bool:
    if not name:
        return True
    low = name
    return any(p in low for p in _PLUMBING)


def _py_label(ev: dict) -> str:
    """python_function names look like 'modeling_qwen3.py(412): Qwen3DecoderLayer.forward'."""
    return ev.get("name", "?").strip()


def _filter_python_frames(raw: list, keep_dist: bool) -> list:
    """Drop plumbing frames; keep the meaningful user/model/c10d frames.

    For NCCL kernels we also keep torch.distributed frames (they ARE the
    'torch 算子' for collectives)."""
    kept = []
    for ev in raw:
        name = _py_label(ev)
        if _is_plumbing(name):
            continue
        if not keep_dist and ("/distributed" in name or "distributed_c10d" in name
                              or "c10d" in name):
            # for non-NCCL kernels, distributed frames are noise
            continue
        kept.append(ev)
    # collapse consecutive frames from the same file:line (dups from reentry)
    out = []
    for ev in kept:
        if out and out[-1].get("name") == ev.get("name"):
            continue
        out.append(ev)
    return out


def _aten_shapes(ev: dict):
    """Pull a compact shape snippet from a cpu_op event."""
    args = ev.get("args", {})
    dims = args.get("Input Dims")
    if dims:
        return dims
    concrete = args.get("Concrete Inputs")
    if concrete:
        return concrete
    return None


def _kernel_detail(ev: dict) -> dict:
    args = ev.get("args", {})
    return {
        "grid": args.get("grid"),
        "block": args.get("block"),
        "smem": args.get("shared memory"),
        "registers": args.get("registers per thread"),
        "device": args.get("device"),
        "cuda_time": ev.get("dur"),
    }


# ─── The builder ──────────────────────────────────────────────────────


def _export_trace(prof) -> dict:
    """Export the profiler chrome trace to a dict (temp file round-trip)."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="nccl_ptx_trace_")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            return json.load(f)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_chains_from_prof(prof, nccl_only: bool = False,
                           verbose: bool = True) -> List[CallChain]:
    """Build one CallChain per unique kernel that ran during the traced region.

    Args:
        prof: the torch.profiler.profile object AFTER its context exited.
        nccl_only: if True, only keep NCCL kernels.
        verbose: print progress.
    """
    trace = _export_trace(prof)
    events = trace.get("traceEvents", [])

    cpu_ops = [e for e in events if e.get("cat") == "cpu_op" and e.get("ph") == "X"]
    py_frames = [e for e in events if e.get("cat") == "python_function" and e.get("ph") == "X"]
    cuda_rt = [e for e in events if e.get("cat") == "cuda_runtime"]
    kernels = [e for e in events if e.get("cat") == "kernel"]

    rt_by_corr = {}
    for e in cuda_rt:
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            rt_by_corr.setdefault(corr, e)

    cpu_idx = _IntervalIndex(cpu_ops)
    py_idx = _IntervalIndex(py_frames)

    # dedup per unique kernel name
    by_name = defaultdict(list)
    for k in kernels:
        by_name[k["name"]].append(k)

    if verbose:
        print(f"  [chain] trace: {len(cpu_ops)} cpu_ops, {len(py_frames)} py frames, "
              f"{len(cuda_rt)} cuda_runtime, {len(kernels)} kernel launches "
              f"({len(by_name)} unique)")

    chains = []
    for prof_name, k_events in by_name.items():
        if nccl_only and not is_nccl_kernel(prof_name):
            continue

        # primary = highest-duration occurrence (most representative)
        primary = max(k_events, key=lambda e: e.get("dur", 0))
        corr = primary.get("args", {}).get("correlation")
        rt = rt_by_corr.get(corr)

        # kernel → cuda_runtime → aten cpu_ops (containment on rt.ts)
        aten_chain = []
        if rt:
            aten_chain = cpu_idx.containing(rt["ts"])

        # aten → python frames (containment on innermost aten ts)
        python_chain = []
        if aten_chain:
            target_ts = aten_chain[-1]["ts"]
            keep_dist = is_nccl_kernel(prof_name)
            python_chain = _filter_python_frames(
                py_idx.containing(target_ts), keep_dist=keep_dist
            )

        # representative callers for shared kernels
        repr_callers = _representative_callers(k_events, rt_by_corr, cpu_idx)

        chain = CallChain(
            kernel_profiler_name=prof_name,
            category=classify_kernel(prof_name) or "Other",
            frames=[],
            occurrence_count=len(k_events),
            total_cuda_time_us=float(sum(e.get("dur", 0) for e in k_events)),
            representative_callers=repr_callers,
        )

        # assemble top → bottom
        for pf in python_chain:
            chain.frames.append(Frame(LAYER_PYTHON, _py_label(pf), source="observed"))
        for co in aten_chain:
            chain.frames.append(Frame(
                LAYER_ATEN, co["name"],
                detail={"shapes": _aten_shapes(co)}, source="observed"))
        # static runtime layers (NCCL/cuBLAS/…)
        chain.add_runtime_layers(
            runtime_knowledge.lookup(prof_name, aten_chain, python_chain))
        if rt:
            chain.frames.append(Frame(
                LAYER_LAUNCH, rt["name"],
                detail={"corr": corr}, source="observed"))
        chain.frames.append(Frame(
            LAYER_KERNEL, prof_name,
            detail=_kernel_detail(primary), source="observed"))

        chains.append(chain)

    # sort: NCCL first (by cuda time), then others by cuda time
    chains.sort(key=lambda c: (0 if c.category.startswith("NCCL") else 1,
                               -c.total_cuda_time_us))
    return chains


def _representative_callers(k_events: list, rt_by_corr: dict,
                            cpu_idx: _IntervalIndex, cap: int = 5) -> List[Caller]:
    """Collect up to `cap` distinct (aten op, shapes) pairs across launches."""
    seen = set()
    out = []
    for k in k_events:
        corr = k.get("args", {}).get("correlation")
        rt = rt_by_corr.get(corr)
        if not rt:
            continue
        aten = cpu_idx.containing(rt["ts"])
        if not aten:
            continue
        aten_name = aten[-1]["name"]
        shapes = _aten_shapes(aten[-1])
        key = (aten_name, json.dumps(shapes, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append(Caller(aten_name, shapes))
        if len(out) >= cap:
            break
    return out


def attach_runtime_category(chains: List[CallChain]):
    """Ensure NCCL kernels get an NCCL category even if classify missed it."""
    for c in chains:
        if is_nccl_kernel(c.kernel_profiler_name) and not c.category.startswith("NCCL"):
            c.category = "NCCL"

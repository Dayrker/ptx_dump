#!/usr/bin/env python3
"""
call_tracer.py — Trace call chains from torch ops → ATen → CUDA kernels → PTX.

Uses:
  - TorchDispatchMode to intercept ATen operator calls
  - torch.profiler to capture CUDA kernel events
  - Post-hoc correlation to build the full chain tree

Output: structured call chain tree showing how each torch op
        reaches its underlying CUDA kernel / NCCL function.
"""

import os
import json
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.profiler import profile, ProfilerActivity


# ─── ATen dispatch tracer ────────────────────────────────────────────


class ATenCallTracer(TorchDispatchMode):
    """
    Intercepts every ATen op via __torch_dispatch__.

    Records:
      - op name (e.g., aten::mm, aten::addmm)
      - input/output shapes
      - timestamp
      - nesting depth (for tree reconstruction)
    """

    def __init__(self):
        self.log = []
        self._depth = 0
        self._stack = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        entry = {
            "op": str(func),
            "op_name": func.name() if hasattr(func, "name") else str(func),
            "timestamp": time.perf_counter(),
            "depth": self._depth,
            "input_shapes": _extract_shapes(args),
            "output_shapes": None,
            "children": [],
        }

        parent = self._stack[-1] if self._stack else None
        self._stack.append(entry)
        self._depth += 1

        try:
            result = func(*args, **kwargs)
        finally:
            self._depth -= 1
            self._stack.pop()

        entry["output_shapes"] = _extract_shapes(result)

        if parent:
            parent["children"].append(entry)
        else:
            self.log.append(entry)

        return result


def _extract_shapes(obj) -> list:
    """Extract tensor shapes from args/result."""
    if isinstance(obj, torch.Tensor):
        return [list(obj.shape)]
    elif isinstance(obj, (list, tuple)):
        shapes = []
        for x in obj:
            if isinstance(x, torch.Tensor):
                shapes.append(list(x.shape))
        return shapes if shapes else None
    return None


# ─── Profiler-based kernel tracer ─────────────────────────────────────


class KernelTracer:
    """
    Uses torch.profiler to capture CUDA kernel launches
    and correlate them with CPU-side ATen ops via key_averages.
    """

    def __init__(self):
        self.key_averages = []  # grouped by op name
        self.cuda_kernels = []  # kernel-only events
        self._profiler = None

    @contextmanager
    def trace(self, activities=None):
        """Context manager for profiling."""
        if activities is None:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            self._profiler = prof
            yield prof

        self._parse_events(prof)

    def _parse_events(self, prof):
        """Parse profiler events using key_averages for proper grouping."""
        # key_averages: groups events by (op name, input shapes)
        for evt in prof.key_averages():
            cuda_time = getattr(evt, "device_time_total", 0)
            record = {
                "name": evt.key,
                "cpu_time_us": evt.cpu_time_total,
                "cuda_time_us": cuda_time,
                "count": evt.count,
                "is_cuda_kernel": not evt.key.startswith("aten::") and cuda_time > 0,
                "is_aten_op": evt.key.startswith("aten::"),
                "input_shapes": getattr(evt, "input_shapes", None),
            }
            self.key_averages.append(record)

            # Separate CUDA kernels
            if record["is_cuda_kernel"]:
                self.cuda_kernels.append(record)


# ─── Call chain builder ───────────────────────────────────────────────


class CallChainBuilder:
    """
    Combines ATen dispatch trace + profiler key_averages
    to build the full call chain: torch.op → ATen → CUDA kernel → PTX.
    """

    def __init__(self):
        self.chains = []
        self.kernel_summary = []

    def build_from_traces(self, aten_log: list, key_averages: list,
                          cuda_kernels: list) -> dict:
        """
        Build call chains.

        Returns dict with:
          - aten_ops: list of ATen op records with frequency
          - kernel_summary: top CUDA kernels by time
          - op_kernel_map: mapping from ATen ops to their kernels
        """
        # ATen ops from dispatch trace
        op_counts = defaultdict(int)
        op_shapes = defaultdict(list)
        flat = []
        self._flatten_aten(aten_log, flat)
        for entry in flat:
            name = entry["op_name"]
            op_counts[name] += 1
            if entry.get("input_shapes"):
                op_shapes[name].append(entry["input_shapes"])

        aten_ops = [
            {"name": name, "count": count, "sample_shapes": op_shapes[name][:3]}
            for name, count in sorted(op_counts.items(), key=lambda x: -x[1])
        ]

        # CUDA kernel summary from profiler
        kernel_summary = sorted(
            [k for k in key_averages if k.get("is_cuda_kernel")],
            key=lambda x: -x["cuda_time_us"],
        )

        # ATen ops from profiler (with CUDA time — these launched kernels)
        aten_with_cuda = [
            k for k in key_averages
            if k.get("is_aten_op") and k["cuda_time_us"] > 0
        ]

        self.chains = aten_ops
        self.kernel_summary = kernel_summary

        return {
            "aten_ops": aten_ops,
            "kernel_summary": kernel_summary,
            "aten_with_cuda": aten_with_cuda,
        }

    def _flatten_aten(self, log: list, flat: list):
        """Flatten nested ATen log into a time-ordered list."""
        for entry in log:
            flat.append(entry)
            if entry.get("children"):
                self._flatten_aten(entry["children"], flat)


# ─── Call chain formatter ─────────────────────────────────────────────


def format_call_chain(result: dict, output_path: str, title: str = "Call Chains"):
    """Write call chains to a human-readable file."""
    aten_ops = result.get("aten_ops", [])
    kernel_summary = result.get("kernel_summary", [])
    aten_with_cuda = result.get("aten_with_cuda", [])

    lines = [
        "=" * 72,
        f"  {title}",
        "=" * 72,
        "",
    ]

    # ─── Section 1: ATen Operation Frequency ───
    lines.append("━" * 72)
    lines.append("  1. ATen Operations (from TorchDispatchMode)")
    lines.append(f"     Total unique ops: {len(aten_ops)}")
    lines.append(f"     Total op calls: {sum(o['count'] for o in aten_ops)}")
    lines.append("━" * 72)
    lines.append("")

    lines.append(f"  {'Count':>6s}  {'Operation':<45s}  Sample Shape")
    lines.append(f"  {'─' * 6}  {'─' * 45}  {'─' * 25}")
    for op in aten_ops:
        shape_str = ""
        if op["sample_shapes"]:
            shape_str = str(op["sample_shapes"][0])
        lines.append(f"  {op['count']:>6d}  {op['name']:<45s}  {shape_str}")
    lines.append("")

    # ─── Section 2: CUDA Kernel Summary ───
    lines.append("━" * 72)
    lines.append("  2. CUDA Kernels (from torch.profiler)")
    lines.append(f"     Total unique kernels: {len(kernel_summary)}")
    lines.append("━" * 72)
    lines.append("")

    if kernel_summary:
        lines.append(f"  {'CUDA Time':>10s}  {'Count':>6s}  {'Kernel':<50s}")
        lines.append(f"  {'─' * 10}  {'─' * 6}  {'─' * 50}")
        for k in kernel_summary[:50]:
            time_str = f"{k['cuda_time_us'] / 1000:.1f}ms"
            lines.append(f"  {time_str:>10s}  {k['count']:>6d}  {k['name'][:50]}")
        if len(kernel_summary) > 50:
            lines.append(f"  ... +{len(kernel_summary) - 50} more kernels")
    else:
        lines.append("  (no CUDA kernels captured)")
    lines.append("")

    # ─── Section 3: ATen → CUDA mapping ───
    lines.append("━" * 72)
    lines.append("  3. ATen → CUDA Mapping (ops that launched kernels)")
    lines.append("━" * 72)
    lines.append("")

    if aten_with_cuda:
        for op in sorted(aten_with_cuda, key=lambda x: -x["cuda_time_us"])[:30]:
            time_str = f"{op['cuda_time_us'] / 1000:.1f}ms"
            lines.append(f"  {op['name']}  ({op['count']}×, CUDA: {time_str})")
            if op.get("input_shapes"):
                lines.append(f"    shapes: {op['input_shapes']}")
            lines.append("")

    # ─── Section 4: NCCL Communication ───
    nccl_kernels = [k for k in kernel_summary if "nccl" in k["name"].lower()]
    if nccl_kernels:
        lines.append("━" * 72)
        lines.append("  4. NCCL Communication Kernels")
        lines.append("━" * 72)
        lines.append("")

        for k in nccl_kernels:
            time_str = f"{k['cuda_time_us'] / 1000:.1f}ms"
            lines.append(f"  {k['name']}")
            lines.append(f"    CUDA time: {time_str} | count: {k['count']}")
            lines.append(f"    Call chain:")
            lines.append(f"      torch.distributed.all_reduce()")
            lines.append(f"        → ProcessGroupNCCL::allreduce()")
            lines.append(f"          → ncclAllReduce()")
            lines.append(f"            → {k['name']}")
            lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


# ─── Convenience: combined tracer ─────────────────────────────────────


@contextmanager
def full_trace_context(trace_aten: bool = True, trace_kernels: bool = True):
    """
    Combined context manager for full call chain tracing.

    Usage:
        with full_trace_context() as tracer:
            model(inputs)
        tracer.write_report("output_dir")
    """
    tracer = FullTracer(trace_aten=trace_aten, trace_kernels=trace_kernels)
    tracer.start()
    try:
        yield tracer
    finally:
        tracer.stop()


class FullTracer:
    """Combined ATen + profiler tracer."""

    def __init__(self, trace_aten: bool = True, trace_kernels: bool = True):
        self.trace_aten = trace_aten
        self.trace_kernels = trace_kernels

        self._aten_tracer = ATenCallTracer() if trace_aten else None
        self._kernel_tracer = KernelTracer() if trace_kernels else None
        self._chain_builder = CallChainBuilder()
        self._result = None

    def start(self):
        if self._aten_tracer:
            self._aten_tracer.__enter__()
        if self._kernel_tracer:
            self._kernel_tracer._ctx = self._kernel_tracer.trace()
            self._kernel_tracer._ctx.__enter__()

    def stop(self):
        if self._aten_tracer:
            self._aten_tracer.__exit__(None, None, None)
        if self._kernel_tracer and hasattr(self._kernel_tracer, '_ctx'):
            self._kernel_tracer._ctx.__exit__(None, None, None)

    def build_chains(self) -> dict:
        """Build call chains from collected traces."""
        aten_log = self._aten_tracer.log if self._aten_tracer else []
        key_avgs = self._kernel_tracer.key_averages if self._kernel_tracer else []
        cuda_kernels = self._kernel_tracer.cuda_kernels if self._kernel_tracer else []

        self._result = self._chain_builder.build_from_traces(
            aten_log, key_avgs, cuda_kernels
        )
        return self._result

    def get_used_kernel_names(self) -> set:
        """Return set of demangled CUDA kernel names captured by profiler."""
        if self._result is None:
            self.build_chains()
        return {k["name"] for k in self._result.get("kernel_summary", [])}

    def write_report(self, output_dir: str, title: str = "Call Chain Report") -> str:
        """Write complete call chain report."""
        if self._result is None:
            self.build_chains()

        report_path = os.path.join(output_dir, "CALL_CHAINS.txt")
        format_call_chain(self._result, report_path, title=title)

        # Also write raw JSON
        json_path = os.path.join(output_dir, "call_chains.json")
        _write_chain_json(self._result, json_path)

        return report_path


def _write_chain_json(result: dict, path: str):
    """Write chains as JSON."""
    serializable = {
        "aten_ops": result.get("aten_ops", [])[:100],
        "kernel_summary": [
            {"name": k["name"], "cuda_time_us": k["cuda_time_us"], "count": k["count"]}
            for k in result.get("kernel_summary", [])[:100]
        ],
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

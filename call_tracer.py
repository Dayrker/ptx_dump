#!/usr/bin/env python3
"""
call_tracer.py — Build per-kernel call chains for the dumped PTX.

This module owns the profiler lifecycle and the report writing. The actual
chain reconstruction (profiler chrome trace → torch op → ATen → runtime → kernel)
lives in chain_builder.py; the static runtime layers live in
runtime_knowledge.py; the data model lives in chain_model.py.

Key correctness point: profiling is scoped to the REAL inference run only —
warmup runs outside the profile so warmup kernels don't pollute the chains.

Flow (in the run scripts):

    with full_trace_context() as tracer:
        # warmup — NOT traced
        _ = model.generate(..., max_new_tokens=5)
        torch.cuda.synchronize()
        # trace ONLY the real run
        with tracer.trace():
            output_ids = model.generate(..., max_new_tokens=N)
        torch.cuda.synchronize()

    chains = tracer.build_chains(nccl_only=args.nccl_only)
    dump_*_ptx(config, ..., chains=chains, used_kernels=...)   # matches chains → PTX
    tracer.write_report(output_dir, chains=chains, nccl_only=...)  # CALL_CHAINS + json
"""

from __future__ import annotations

import os
import json
from contextlib import contextmanager
from typing import List, Optional

from torch.profiler import profile, ProfilerActivity

from chain_model import CallChain, chain_to_jsonable
from chain_builder import build_chains_from_prof
from symbol_utils import is_nccl_kernel


class FullTracer:
    """Owns a torch.profiler profile scoped to the real run, plus chain output."""

    def __init__(self):
        self._prof: Optional[profile] = None
        self._chains: List[CallChain] = []
        self._tracing = False

    @contextmanager
    def trace(self):
        """Context manager that profiles ONLY the code inside it.

        Put the warmup OUTSIDE this block so warmup kernels are excluded from
        the chains."""
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,            # needed for python_function events
        )
        self._prof.__enter__()
        self._tracing = True
        try:
            yield self._prof
        finally:
            self._prof.__exit__(None, None, None)
            self._tracing = False

    def build_chains(self, nccl_only: bool = False) -> List[CallChain]:
        """Reconstruct one CallChain per unique kernel from the profile."""
        if self._prof is None:
            return []
        self._chains = build_chains_from_prof(self._prof, nccl_only=nccl_only)
        return self._chains

    def get_used_kernel_names(self) -> set:
        """Kernel names that actually ran (for used-only PTX filtering)."""
        if not self._chains and self._prof is not None:
            self.build_chains()
        return {c.kernel_profiler_name for c in self._chains}

    def write_report(self, output_dir: str, chains: Optional[List[CallChain]] = None,
                     nccl_only: bool = False,
                     title: str = "Call Chains") -> str:
        """Write CALL_CHAINS.txt (human-readable) + call_chains.json."""
        if chains is None:
            chains = self._chains
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "CALL_CHAINS.txt")
        with open(report_path, "w") as f:
            f.write(self._render_report(chains, title=title, nccl_only=nccl_only))

        # JSON — keyed by mangled name (or profiler name if unmatched)
        json_path = os.path.join(output_dir, "call_chains.json")
        serial = {}
        for c in chains:
            key = c.kernel_mangled_name or c.kernel_profiler_name
            serial[key] = chain_to_jsonable(c)
        with open(json_path, "w") as f:
            json.dump(serial, f, indent=2, default=str)

        return report_path

    # ── internal ──

    def _render_report(self, chains: List[CallChain], title: str,
                       nccl_only: bool) -> str:
        lines = [
            "=" * 78,
            f"  {title}",
            "=" * 78,
            "",
            f"  Unique kernels traced : {len(chains)}",
            f"  NCCL-only filter      : {nccl_only}",
            f"  Profiler              : torch.profiler chrome trace "
            f"(python_function → cpu_op → cuda_runtime → kernel, joined by correlation)",
            f"  Runtime layers        : static knowledge (runtime_knowledge.py)",
            "",
        ]

        matched = [c for c in chains if c.matched]
        unmatched = [c for c in chains if not c.matched]

        if nccl_only and not any(is_nccl_kernel(c.kernel_profiler_name) for c in chains):
            lines.append("  ! No NCCL kernels observed in this run.")
            lines.append("    (possible causes: tiny tensors took the single-rank memcpy")
            lines.append("     fast path, or NCCL was not initialized.)")
            lines.append("")

        # NCCL kernels first, then the rest
        nccl_chains = [c for c in matched if c.category.startswith("NCCL")]
        other_chains = [c for c in matched if not c.category.startswith("NCCL")]

        if nccl_chains:
            lines.append("━" * 78)
            lines.append(f"  NCCL kernels with PTX ({len(nccl_chains)})")
            lines.append("━" * 78)
            for c in nccl_chains:
                lines.append(c.render_block())
                lines.append("")

        if other_chains:
            lines.append("━" * 78)
            lines.append(f"  Other kernels with PTX ({len(other_chains)})")
            lines.append("━" * 78)
            for c in other_chains:
                lines.append(c.render_block())
                lines.append("")

        if unmatched:
            lines.append("━" * 78)
            lines.append(f"  Kernels with NO matched PTX ({len(unmatched)})")
            lines.append("  (ran during inference but not present in the dumped libs)")
            lines.append("━" * 78)
            for c in unmatched[:50]:
                lines.append(f"  • {c.kernel_profiler_name[:70]}")
                lines.append(f"      {c.match_note or 'no PTX block matched'}")
                lines.append(f"      category={c.category}  calls={c.occurrence_count}")
            if len(unmatched) > 50:
                lines.append(f"  ... +{len(unmatched) - 50} more")
            lines.append("")

        lines.append("=" * 78)
        return "\n".join(lines)


@contextmanager
def full_trace_context(trace_aten: bool = True, trace_kernels: bool = True):
    """Combined context manager. Profiling is started later via tracer.trace()."""
    # trace_aten / trace_kernels kept for API compatibility; the chrome trace
    # captures both unconditionally inside tracer.trace().
    tracer = FullTracer()
    try:
        yield tracer
    finally:
        pass

#!/usr/bin/env python3
"""
runtime_knowledge.py — Static "runtime layer" knowledge for call chains.

The profiler observes: the Python torch op, the ATen op, the cudaLaunchKernel
call, and the device kernel. It does NOT observe the intermediate C++ runtime
functions between ATen and the kernel launch (ProcessGroupNCCL::allreduce,
ncclAllReduce, ncclEnqueueCheck, cublasGemmEx, …). Those layers are *static
knowledge* — deterministic per (aten op / kernel family) — and we encode them
here so the chain reads "一路链接都写上" (every hop written out).

The NCCL paths are sourced from docs/allreduce-deep-dive.md (the 9-layer
AllReduce walkthrough). cuBLAS / cuDNN paths are standard PyTorch knowledge.

This is a data-driven table: add a RuntimeRule, no code changes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import List, Optional

from nccl_ptx_lib.core.chain_model import Frame, LAYER_RUNTIME
from nccl_ptx_lib.core.symbol_utils import is_nccl_kernel, classify_kernel


@dataclass
class RuntimeLayer:
    name: str            # human-readable, e.g. "ProcessGroupNCCL::allreduce()"
    symbol: str = ""     # optional mangled/C symbol hint
    notes: str = ""


@dataclass
class RuntimeRule:
    """First match wins. An empty match dict matches everything (fallback)."""
    kernel_glob: str = ""    # fnmatch over the profiler kernel name
    aten_glob: str = ""      # fnmatch over the innermost aten op name
    layers: List[RuntimeLayer] = field(default_factory=list)
    category: str = "Other"


# ─── NCCL collective kernels ───────────────────────────────────────────
# ncclDevKernel_<Coll>_<RedOp>_<dtype>_<ALGO>_<PROTO>  (see allreduce-deep-dive.md)
# The full device path: ncclKernelMain → RunWork<Coll,…>::run() → run<Algo><Proto>()

def _nccl_common_tail(coll_label: str, dev_layer: str) -> List[RuntimeLayer]:
    return [
        RuntimeLayer("ncclEnqueueCheck()",
                     notes="CommCheck/ArgsCheck; taskAppend; hostToDevRedOp (ncclSum→ncclDevSum)"),
        RuntimeLayer("taskAppend()",
                     notes="join group; sort-into collQueue"),
        RuntimeLayer("scheduleCollTasksToPlan()",
                     notes="topoGetAlgoInfo brute-force min-time (RING/TREE/NVLS); ncclDevFuncId; initCollWorkElem; uploadWork"),
        RuntimeLayer("ncclLaunchKernel()",
                     notes="grid={channelCount}; block={LL≤512, tuned≥96}; smem=ncclShmemDynamicSize(); cudaLaunchKernelExC"),
        RuntimeLayer("ncclKernelMain<…>()  [device entry]",
                     notes="blockIdx→channelId; load devComm/channel/work → shared mem"),
        RuntimeLayer(dev_layer,
                     notes=f"RunWork<{coll_label},…>::run() → device-side collective"),
    ]


def _algo_from_kernel(kernel_name: str) -> str:
    """Recover the NCCL algo from the specialized kernel name suffix
    (ncclDevKernel_AllReduce_Sum_f16_RING_LL → 'RING')."""
    n = (kernel_name or "").upper()
    for algo in ("RING", "TREE", "NVLS", "COLLNET_DIRECT", "COLLNET_CHAIN"):
        if f"_{algo}_" in n or n.endswith(f"_{algo}"):
            return algo
    return ""


KNOWLEDGE: List[RuntimeRule] = [
    # ── AllReduce via dist.barrier() ──
    # NCCL has NO barrier collective; dist.barrier() reuses the AllReduce
    # machinery on a cached 1-element barrierTensor_. Only the top two layers
    # differ from dist.all_reduce(); Layer 4+ (ncclAllReduce …) is identical.
    # Must come BEFORE the generic AllReduce rule (aten_glob is more specific).
    RuntimeRule(
        kernel_glob="ncclDevKernel_AllReduce_*",
        aten_glob="c10d::barrier*",
        category="NCCL/AllReduce",
        layers=[
            RuntimeLayer("torch.distributed.barrier()", notes="Python c10d entry (distributed_c10d.py:4122)"),
            RuntimeLayer("ProcessGroupNCCL::barrier()", notes="reuses AllReduce on cached 1-elem barrierTensor_ (header:610/358)"),
            RuntimeLayer("ProcessGroupNCCL::allreduce_impl() → collective()", notes="fn lambda wraps ncclAllReduce"),
            RuntimeLayer("ncclAllReduce()  [NCCL C API]", symbol="ncclAllReduce",
                         notes="builds ncclInfo{ncclFuncAllReduce} on barrierTensor_"),
            *_nccl_common_tail("AllReduce", "RunWork<AllReduce,…>::run() → run{Algo}<Proto>"),
        ],
    ),
    # ── AllReduce via dist.all_reduce() ──
    RuntimeRule(
        kernel_glob="ncclDevKernel_AllReduce_*",
        category="NCCL/AllReduce",
        layers=[
            RuntimeLayer("torch.distributed.all_reduce()", notes="Python c10d entry; creates output tensor"),
            RuntimeLayer("ProcessGroupNCCL::allreduce()", symbol="_ZNK17ProcessGroupNCCL9allreduceE…",
                         notes="complex→view_as_real; intraNode fast-path bypass"),
            RuntimeLayer("ProcessGroupNCCL::allreduce_impl()", notes="builds the ncclAllReduce lambda"),
            RuntimeLayer("ProcessGroupNCCL::collective()", notes="getNCCLComm; syncStream; Work obj; calls fn lambda"),
            RuntimeLayer("ncclAllReduce()  [NCCL C API]", symbol="ncclAllReduce",
                         notes="builds ncclInfo{ncclFuncAllReduce}"),
            *_nccl_common_tail("AllReduce", "RunWork<AllReduce,…>::run() → run{Algo}<Proto>"),
        ],
    ),
    # ── AllGather ──
    RuntimeRule(
        kernel_glob="ncclDevKernel_AllGather_*",
        category="NCCL/AllGather",
        layers=[
            RuntimeLayer("torch.distributed.all_gather() / _all_gather_base()"),
            RuntimeLayer("ProcessGroupNCCL::allgather()"),
            RuntimeLayer("collective() → ncclAllGather()", symbol="ncclAllGather"),
            *_nccl_common_tail("AllGather", "RunWork<AllGather,…>::run()"),
        ],
    ),
    # ── ReduceScatter ──
    RuntimeRule(
        kernel_glob="ncclDevKernel_ReduceScatter_*",
        category="NCCL/ReduceScatter",
        layers=[
            RuntimeLayer("torch.distributed.reduce_scatter()"),
            RuntimeLayer("ProcessGroupNCCL::reduce_scatter()"),
            RuntimeLayer("collective() → ncclReduceScatter()", symbol="ncclReduceScatter"),
            *_nccl_common_tail("ReduceScatter", "RunWork<ReduceScatter,…>::run()"),
        ],
    ),
    # ── Broadcast ──
    RuntimeRule(
        kernel_glob="ncclDevKernel_Broadcast_*",
        category="NCCL/Broadcast",
        layers=[
            RuntimeLayer("torch.distributed.broadcast()"),
            RuntimeLayer("ProcessGroupNCCL::broadcast()"),
            RuntimeLayer("collective() → ncclBroadcast()", symbol="ncclBroadcast"),
            *_nccl_common_tail("Broadcast", "RunWork<Broadcast,…>::run()"),
        ],
    ),
    # ── Send / Recv / P2P ──
    RuntimeRule(
        kernel_glob="ncclDevKernel_*Send*",
        category="NCCL/Send",
        layers=[
            RuntimeLayer("torch.distributed.send() / P2P"),
            RuntimeLayer("ProcessGroupNCCL::send() → ncclSend()", symbol="ncclSend"),
            RuntimeLayer("ncclLaunchKernel() → ncclKernelMain → RunWork<P2p,…>::run()"),
        ],
    ),
    # ── Generic NCCL (unmatched collective) ──
    RuntimeRule(
        kernel_glob="ncclDevKernel_*",
        category="NCCL",
        layers=[
            RuntimeLayer("torch.distributed.*  [c10d collective]"),
            RuntimeLayer("ProcessGroupNCCL::*() → collective()"),
            RuntimeLayer("nccl*()  [NCCL C API] → ncclEnqueueCheck → taskAppend"),
            RuntimeLayer("scheduleCollTasksToPlan() → ncclLaunchKernel() → ncclKernelMain → RunWork<…>::run()"),
        ],
    ),
    # ── cuBLAS GEMM ──
    RuntimeRule(
        kernel_glob="*gemm*",
        aten_glob="aten::mm|aten::addmm|aten::bmm|aten::matmul|aten::linear|aten::addmm",
        category="cuBLAS",
        layers=[
            RuntimeLayer("at::native::mm / addmm / bmm", notes="dispatches to cuBLAS via THBlas"),
            RuntimeLayer("cublasGemmEx() / cublasGemmStridedBatchedEx()", symbol="cublasGemmEx",
                         notes="cuBLAS C API; algo picked by heuristic"),
            RuntimeLayer("ampere_*gemm_*  [cuBLAS device kernel]", notes="launched by cuBLAS internal helper"),
        ],
    ),
    # ── cuDNN scaled-dot-product attention ──
    RuntimeRule(
        kernel_glob="*flash*attention*",
        aten_glob="aten::_scaled_dot_product*",
        category="cuDNN",
        layers=[
            RuntimeLayer("at::native::_scaled_dot_product_flash_attention"),
            RuntimeLayer("cudnn::sdpa::run()  [cuDNN frontend graph]", notes="flash-attention kernels"),
        ],
    ),
    RuntimeRule(
        kernel_glob="*scaled_dot_product*",
        aten_glob="aten::_scaled_dot_product*",
        category="cuDNN",
        layers=[
            RuntimeLayer("at::native::_scaled_dot_product_attention"),
            RuntimeLayer("cudnn / flash attn backend"),
        ],
    ),
    # ── cuDNN (generic) ──
    RuntimeRule(
        kernel_glob="*cudnn*",
        category="cuDNN",
        layers=[RuntimeLayer("cudnn::*()  [cuDNN runtime]")],
    ),
    # ── cuRAND (random kernels) ──
    RuntimeRule(
        kernel_glob="*distribution*|*curand*|*Philox*",
        category="cuRAND",
        layers=[
            RuntimeLayer("at::native::* (distribution kernel)", notes="Philox RNG state"),
            RuntimeLayer("curand* / at::native distribution kernel"),
        ],
    ),
    # ── Elementwise / reductions (ATen native) ──
    RuntimeRule(
        kernel_glob="*elementwise*|*reduce_kernel*|*unrolled_elementwise*",
        category="ATen",
        layers=[
            RuntimeLayer("at::native::elementwise_kernel / reduce_kernel",
                         notes="fused elementwise / reduction; launched directly by the aten op"),
        ],
    ),
    # ── memcpy / memset ──
    RuntimeRule(
        kernel_glob="*memcpy*|*memset*|*cudaMemcpy*",
        category="MemOps",
        layers=[RuntimeLayer("cudaMemcpyAsync / cudaMemsetAsync  [runtime]")],
    ),
    # ── Fallback (must be last) ──
    RuntimeRule(
        kernel_glob="",
        category="Other",
        layers=[
            RuntimeLayer("at::native::<op>  [generic ATen dispatch]",
                         notes="no static runtime path known for this kernel family"),
        ],
    ),
]


def _matches(rule: RuntimeRule, kernel_name: str, aten_name: str,
             aten_chain_names: list = None) -> bool:
    if rule.kernel_glob and not _any_match(rule.kernel_glob, kernel_name):
        return False
    if rule.aten_glob:
        # A rule with an aten_glob matches if ANY aten op in the launch chain
        # matches (not just the innermost) — e.g. c10d::barrier is an outer
        # op; record_param_comms may be the innermost.
        names = [aten_name]
        if aten_chain_names:
            names = list(aten_chain_names) + [aten_name]
        if not any(_any_match(rule.aten_glob, n) for n in names if n):
            return False
    return True


def _any_match(globs: str, name: str) -> bool:
    name = name or ""
    return any(fnmatchcase(name, g.strip()) for g in globs.split("|") if g.strip())


def lookup(kernel_profiler_name: str,
           aten_chain: list,
           python_chain: list) -> List[Frame]:
    """Return the static runtime Frame layers for a kernel.

    aten_chain: list of cpu_op dicts (innermost last) that launched the kernel.
    python_chain: list of python_function dicts (unused for matching, reserved).
    """
    aten_chain_names = [e.get("name", "") for e in aten_chain] if aten_chain else []
    aten_name = aten_chain_names[-1] if aten_chain_names else ""
    kernel_name = kernel_profiler_name or ""

    for rule in KNOWLEDGE:
        if _matches(rule, kernel_name, aten_name, aten_chain_names):
            cat = rule.category
            # refine NCCL category via symbol_utils if the glob was generic
            if not cat.startswith("NCCL") and is_nccl_kernel(kernel_name):
                cat = classify_kernel(kernel_name) or "NCCL"
            frames = [
                Frame(LAYER_RUNTIME, layer.name,
                      detail={"symbol": layer.symbol} if layer.symbol else {},
                      source="static")
                for layer in rule.layers
            ]
            # Specialize the device-dispatch layer for NCCL kernels: the algo
            # is encoded in the kernel name (_RING / _TREE), so replace the
            # 'run{Algo}' placeholder with the concrete runRing/runTree.
            algo = _algo_from_kernel(kernel_name)
            if algo and cat.startswith("NCCL"):
                run_fn = {"RING": "runRing", "TREE": "runTree"}.get(algo, f"run{algo}")
                for f in frames:
                    if "run{Algo}" in f.name:
                        f.name = f.name.replace("run{Algo}", run_fn)
            return frames
    return []

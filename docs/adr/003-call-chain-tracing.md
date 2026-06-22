# ADR-003: 调用链路追踪（torch 算子 → ATen → runtime → kernel → PTX）

## 状态
已更新（2026-06-21，重构为真实的 per-kernel 链路）

## 背景
我们需要为每个 dump 的 PTX kernel 附带一条**真实**的调用链路：从 torch 算子，
经 ATen 调度，到底层 runtime、kernel launch、PTX 文件，每一跳都写出来
（「一路链接都写上」）。

旧实现（`TorchDispatchMode` + `prof.key_averages()`）只能产出 ATen op 频率表
和 CUDA kernel 时间表，NCCL 链路甚至是硬编码字符串模板——不是真实链路。

## 决策
以 `torch.profiler` 的 **chrome trace** 为唯一真源重建链路。

### 工作原理（torch 2.5.1 已验证）
`profile(activities=[CPU,CUDA], record_shapes=True, with_stack=True)` 退出后，
`prof.export_chrome_trace(path)` 产出 JSON，含四类事件、共用一个绝对 µs 时钟：

| cat | 含义 | 关键字段 |
|-----|------|---------|
| `python_function` | torch 算子（Python 帧） | name = `file.py(lineno): func` |
| `cpu_op` | ATen op | `args.Input Dims`（shapes） |
| `cuda_runtime` | `cudaLaunchKernel`/`cudaLaunchKernelExC` | `args.correlation` |
| `kernel` | device kernel | `args.correlation` + grid/block/smem |

链路按 `correlation` + 时间包含关系还原：

```
kernel.correlation  ──match──▶ cuda_runtime (same correlation)
cuda_runtime.ts      ─contains──▶ cpu_op[ts, ts+dur]   (ATen op(s))
innermost cpu_op.ts  ─contains──▶ enclosing python_function 帧
```

见 `nccl_ptx_lib/chain/chain_builder.py`。

### 为什么不用 `FunctionEvent.stack` / `_get_kineto_results`
torch 2.5.1 中 `FunctionEvent.stack` **恒为空**（即使 `with_stack=True`）；
`prof._get_kineto_results()` / `experimental_event_tree()` 在此版本**不存在**。
chrome trace 的 `python_function` 事件才是 Python 层的可靠来源。

### 为什么 runtime 层用静态知识表
profiler 只能看到 torch 算子、ATen op、`cudaLaunchKernel`、device kernel。
ATen 与 kernel 之间的 C++ runtime（`ProcessGroupNCCL::allreduce` →
`ncclAllReduce` → `ncclEnqueueCheck` → … ；`cublasGemmEx` → …）无法观测，
但每个 kernel 族是确定性的。`nccl_ptx_lib/chain/runtime_knowledge.py` 用数据驱动的
`RuntimeRule` 表编码这些层（NCCL 路径源自 `docs/allreduce-deep-dive.md` 的
9 层全链路）。

### warmup / sync 隔离
`FullTracer.trace()` 是一个子 context manager：warmup 放在它**外面**，profiler
只覆盖真实推理 run，避免 warmup kernel 污染链路。双卡模式默认也把
`dist.barrier()` 这类同步 collective 放在 trace 外面：barrier 仍用于 rank
同步，但不计入 used-only dump，因此默认输出代表模型 `generate()` 内部真实用到
的 NCCL kernel。

如需研究“整个程序运行期间的 NCCL 同步”，可加 `--include-sync-kernels`，此时真实
推理后的 barrier 也纳入 profiler/used-only 集合，通常会额外保留一个 count=1
的 NCCL AllReduce kernel。

## 影响
- 每个 per-kernel `.ptx` 文件头部嵌入其调用链块；`CALL_CHAINS.txt` /
  `call_chains.json` 汇总。
- cuBLAS profiler 名称（`ampere_*gemm*`）与 PTX `.entry` mangled 符号不是 1:1，
  按 *族*（gemm/elementwise/reduce…）匹配，见 `nccl_ptx_lib/ptx/ptx_dumper.py` 的 `match_chain_to_ptx`。
- 双卡仅 rank 0 追踪；NCCL kernel 符号跨 rank 一致，PTX 来自本地 `libnccl.so`。

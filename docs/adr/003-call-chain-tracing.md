# ADR-003: 基于 TorchDispatchMode 的调用链追踪

## 状态
已采纳（2026-06-17）

## 背景
我们需要追踪从 `torch.op()` → ATen 调度 → CUDA kernel → PTX 函数的完整路径。PyTorch 自带的 profiler 能提供 kernel 名称，但无法展示调度链路。

## 决策
使用 `TorchDispatchMode`（Python 层的 ATen 调度钩子）配合 `torch.profiler`（CUDA 事件捕获）来实现。

### 工作原理
1. `TorchDispatchMode.__torch_dispatch__` 拦截每一个 ATen 操作调用
2. 记录：操作名称、输入 shape、输出 shape、时间戳、嵌套深度
3. `torch.profiler` 通过 `key_averages()` 捕获 CUDA kernel 事件（含时间统计）
4. 后处理阶段：通过 profiler 自身的分组机制关联 ATen 操作和 CUDA kernel
5. 构建调用树：`torch.matmul` → `aten::mm` → `ampere_sgemm_128x64_tn`

### 为什么不用 `torch.fx`？
`torch.fx` 追踪的是静态计算图 — 无法捕获动态调度和 NCCL 通信调用。`TorchDispatchMode` 能在运行时看到每一个操作，包括分布式操作。

### 为什么不用 `CUDA_LAUNCH_BLOCKING=1` + 堆栈追踪？
将所有 kernel 串行化执行会严重改变时序，且无法捕获异步的 NCCL 操作。我们的方案开销较低（约 10-20%），且保留了真实的执行语义。

### 为什么用 `key_averages()` 做关联？
`prof.key_averages()` 按操作名称分组，已经内置了 CPU 操作和 CUDA kernel 的时间关联。相比手动按时间窗口匹配（时钟源不一致会导致错误），这种方式更可靠。

## 影响
- 追踪开销约 10-20%（对于分析场景可接受）
- 嵌套深度追踪支持调用树重建
- NCCL 调用通过 `dist.all_reduce` 在调度日志中可见

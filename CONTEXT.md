# CONTEXT.md — 共享领域术语

本项目的精简术语表。开发者和 Agent 统一使用这些术语，避免歧义。

## GPU / CUDA

| 术语 | 定义 |
|------|------|
| **PTX** | Parallel Thread eXecution — NVIDIA 的虚拟 GPU 汇编（中间表示）。文本形式，架构无关。由 nvcc 使用 `code=compute_XX` 生成。 |
| **SASS** | Streaming ASSembly — 实际的 GPU 机器码。二进制形式，架构相关。由 nvcc 使用 `code=sm_XX` 生成。 |
| **fatbin** | Fat binary — .so/.cubin 内部的容器，同时包含多架构的 PTX 和 SASS。 |
| **cuobjdump** | CUDA 工具，用于从 fatbin 中提取 PTX/SASS/资源信息。 |
| **kernel** | 在 GPU 上执行的 CUDA 函数。具有 mangled C++ 名称和启动配置（grid/block）。 |
| **warp** | GPU 上 32 个锁步执行的线程。 |
| **occupancy** | 活跃 warp 数与每个 SM 最大 warp 数的比率。受寄存器和共享内存使用量影响。 |
| **SM** | Streaming Multiprocessor — GPU 上的一个处理单元。A100 有 108 个 SM。 |
| **sm_80** | Compute capability 8.0 — A100 架构。 |

## NCCL

| 术语 | 定义 |
|------|------|
| **NCCL** | NVIDIA Collective Communications Library — 高性能多 GPU 通信库。 |
| **AllReduce** | 对所有 rank 的张量求和，将结果广播给所有人。数据并行训练的核心操作。 |
| **Ring** | 环形拓扑 — GPU 以环形传递数据。适合大张量。 |
| **Tree** | 树形拓扑 — 层级归约。适合小张量。 |
| **LL / LL128** | Low-Latency 协议 — 4 字节 / 128 字节粒度。用于小消息。 |
| **Simple** | Simple 协议 — 批量数据传输。用于大消息。 |
| **TP** | Tensor Parallelism — 将模型权重切分到多个 GPU。本地 matmul 后使用 AllReduce 聚合。 |

## PyTorch 调度机制

| 术语 | 定义 |
|------|------|
| **ATen** | PyTorch 的 C++ 张量库。所有 `torch.*` 操作最终调度到 ATen kernel。 |
| **Dispatcher** | PyTorch 的调度组件，负责将 `torch.op()` 路由到注册的 kernel（CPU/CUDA/...）。 |
| **TorchDispatchMode** | Python 上下文管理器，拦截每一个 ATen 操作。用于追踪调用链。 |
| **Profiler** | `torch.profiler` — 记录 CPU 操作和 CUDA kernel 事件（含时间戳）。 |

## 项目专用术语

| 术语 | 定义 |
|------|------|
| **call chain** | 从 `torch.op()` → ATen 调度 → CUDA kernel → PTX 函数的完整路径。我们要追踪的东西。 |
| **dump** | 提取并保存 PTX/SASS 为可读文件。 |
| **demangle** | 通过 `c++filt` 将 C++ mangled 的 kernel 名称还原为可读形式。 |
| **single_ptx/** | 单卡模式的 PTX 输出目录（包含所有 CUDA kernel）。 |
| **nccl_ptx/** | 双卡模式的 PTX 输出目录（包含 NCCL kernel）。 |

## 硬件环境（本机）

| 组件 | 规格 |
|------|------|
| GPU | 8× NVIDIA A100-SXM4-80GB |
| 架构 | sm_80 |
| CUDA | 12.1 |
| NCCL | 2.21.5（本地编译：`/home/zhangchen/PTX/nccl/`） |
| Conda 环境 | `torch251`（PyTorch 2.5.1+cu121） |
| 模型 | Qwen3-8B，位于 `/home/model/Qwen3-8B` |

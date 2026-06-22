# NCCL PTX Dump — Qwen3-8B Inference + PTX Extraction

一键运行 Qwen3-8B 推理并 dump CUDA PTX 汇编，支持单卡和双卡模式。

## 环境

| 组件 | 版本 | 路径 |
|------|------|------|
| PyTorch | 2.5.1+cu121 | `conda activate torch251` |
| CUDA | 12.1 | `/usr/local/cuda-12.1` |
| NCCL | 2.21.5 (本地编译) | `/home/zhangchen/PTX/nccl/` |
| GPU | 8× A100 (sm_80) | — |
| Model | Qwen3-8B | `/home/model/Qwen3-8B` |

## 快速开始

```bash
# 激活环境
conda activate torch251
cd ~/PTX/nccl_ptx_dump

# 单卡推理 + dump 所有用到的 PTX（cuBLAS/Triton/NCCL）+ 调用链路
python run.py single --dump-ptx

# 双卡推理 + dump NCCL PTX（只保留 NCCL 相关）+ 调用链路
python run.py dual --dump-ptx --nccl-only

# 如果也想把同步 collective（例如 dist.barrier）纳入 used-only dump
python run.py dual --dump-ptx --nccl-only --include-sync-kernels

# 双卡 dump 全部 PTX（含非 NCCL kernel）
python run.py dual --dump-ptx
```

> 本地 NCCL 在运行时通过 `LD_PRELOAD` 强制加载（见下「NCCL 本地编译」），
> 因此 dump 出的 PTX 与实际执行的 kernel 一致。

## CLI 参考

```
python run.py <mode> [options]

Modes:
  single    单卡推理，PTX 输出到 single_ptx/
  dual      双卡 TP 推理，PTX 输出到 nccl_ptx/

Options:
  --dump-ptx          开启 PTX dump
  --dump-sass         额外 dump SASS（GPU 机器码），与 PTX 并行输出
  --nccl-only         (dual only) 只保留 NCCL 相关 kernel
  --include-sync-kernels
                      (dual only) used-only 模式也保留同步 collective
                      （如 dist.barrier）触发的 NCCL kernel
  --all-kernels       dump 库中所有 kernel（默认只保留运行时实际用到的 kernel）
  --trace-calls       记录 torch → ATen → CUDA 调用链
  --model-path PATH   模型路径 (默认 /home/model/Qwen3-8B)
  --prompt TEXT       自定义输入 prompt
  --max-new-tokens N  最大生成 token 数
  --output-dir DIR    覆盖输出目录
```

## 输出结构

### 单卡模式 (`single_ptx/`)

```
single_ptx/
├── SUMMARY.txt           # Kernel 汇总 (名称、分类、寄存器使用)
├── single_all.ptx        # 所有 PTX 合并 (带注释)
├── single_001_*.ptx      # 每个 kernel 单独文件
├── CALL_CHAINS.txt       # 调用链路 (torch → ATen → CUDA)
└── call_chains.json      # 调用链路 (JSON 格式)
```

### 双卡模式 (`nccl_ptx/`)

```
nccl_ptx/
├── SUMMARY.txt           # NCCL kernel 汇总
├── nccl_all.ptx          # 所有 NCCL PTX (带注释)
├── nccl_001_*.ptx        # 每个 kernel 单独文件
├── CALL_CHAINS.txt       # 调用链路 (torch → ATen → NCCL)
└── call_chains.json      # 调用链路 (JSON 格式)
```

## 架构

代码按层分包 `nccl_ptx_lib/`（`run.py` 留在根目录做入口；输出目录 `single_ptx/`、
`nccl_ptx/` 也在根目录）：

```
run.py                            # CLI 入口 (统一参数解析 + 调度 + LD_PRELOAD)
nccl_ptx_lib/
├── core/                         # 叶子层（无内部依赖）
│   ├── env_setup.py              # 环境配置 (路径、env vars、LD_PRELOAD、自动探测 cuBLAS/Triton)
│   ├── symbol_utils.py           # 符号解析 (demangle + kernel 分类)
│   └── chain_model.py            # Frame/CallChain 数据模型
├── chain/                        # 调用链子系统（依赖 core）
│   ├── chain_builder.py          # chrome-trace 重建 torch→aten→runtime→kernel 链路
│   ├── runtime_knowledge.py      # 静态 runtime 层知识表 (NCCL/cuBLAS/cuDNN/…，数据驱动)
│   └── call_tracer.py            # profiler 生命周期 + 调用链报告 (CALL_CHAINS.txt/json)
├── ptx/                          # PTX 提取子系统（依赖 core）
│   ├── ptx_dumper.py             # PTX 提取编排 (cuobjdump + JIT/Triton + profiler↔PTX 匹配)
│   └── ptx_formatter.py          # PTX 格式化 + 调用链块嵌入每个 kernel 文件
└── runners/                      # 推理编排（依赖 chain + ptx + core）
    ├── run_single_gpu.py         # 单卡推理 + tracing (cuBLAS/Triton/NCCL PTX)
    └── run_dual_gpu.py           # 双卡 TP 推理 + NCCL tracing (torchrun)
```

依赖方向：`core` ← `{chain, ptx}` ← `runners` ← `run.py`，无环。各 runner 由 `run.py`
以子进程方式启动（双卡走 `torchrun`），runner 顶部会把仓库根加入 `sys.path` 以解析
`nccl_ptx_lib.*` 包导入。

## PTX 可读性增强

- **符号还原**: 所有 C++ mangled name 自动 demangle
- **指令注释**: 每条 PTX 指令标注类别 (LOAD/STORE/ARITH/SYNC...)
- **Kernel 分类**: 自动分为 NCCL/cuBLAS/cuDNN/ATen 等类别
- **分文件输出**: 每个 kernel 单独一个 .ptx 文件
- **汇总表**: SUMMARY.txt 列出所有 kernel 及其元数据

## 调用链路（torch 算子 → ATen → runtime → kernel → PTX）

每个 dump 的 PTX kernel 都附带一条**真实的、逐跳写出**的调用链路。
profiler 事件树在同一时钟上串起四类事件，按 `correlation` + 时间包含关系还原全链路：

```
[python]  modeling_qwen3.py(81): Qwen3MLP.forward → Linear.forward
[aten]    aten::linear → aten::matmul → aten::mm   shapes=[[12,12288],[12288,4096]]
[runtime] at::native::mm → cublasGemmEx() → ampere_*gemm_*    ← 静态知识 (nccl_ptx_lib/chain/runtime_knowledge.py)
[launch]  cudaLaunchKernel  corr=26153
[kernel]  ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_...
[ptx]     single_001_..._.ptx
```

NCCL 的链路覆盖 `docs/allreduce-deep-dive.md` 的全部 9 层：

```
[python] torch.distributed.all_reduce()
[aten]   c10d::allreduce_
[runtime] ProcessGroupNCCL::allreduce → allreduce_impl → collective
        → ncclAllReduce → ncclEnqueueCheck → taskAppend
        → scheduleCollTasksToPlan → ncclLaunchKernel
        → ncclKernelMain → RunWork<AllReduce,…>::run() → runRing<ProtoLL>
[launch] cudaLaunchKernelExC  corr=54244
[kernel] ncclDevKernel_AllReduce_Sum_f16_RING_LL
[ptx]    nccl_003_..._RING_LL_...ptx
```

- **python / aten / launch / kernel / ptx** 层来自 `torch.profiler` chrome trace
  （`python_function` / `cpu_op` / `cuda_runtime` / `kernel` 四类事件）。
- **runtime** 层（ProcessGroupNCCL→ncclAllReduce→…、cublasGemmEx→…）profiler
  看不到，是 `nccl_ptx_lib/chain/runtime_knowledge.py` 里的数据驱动静态知识表。
- 链路写在每个 per-kernel `.ptx` 文件头部，也汇总在 `CALL_CHAINS.txt` /
  `call_chains.json`。
- 「一路链接都写上」指的是：从 torch 算子、ATen 算子、底层 runtime、kernel launch、
  到具体 PTX 文件，每一跳都有对应行。

### 注意事项

- 默认 used-only 只统计真实推理 `model.generate()` 内部触发的 kernel；同步用的
  `dist.barrier()` 仍会执行，但不进入 profiler/dump。加 `--include-sync-kernels`
  可把这些同步 collective 也纳入输出，常见表现是额外出现一个很小的
  `ncclDevKernel_AllReduce_Sum_f32_*`。
- cuBLAS 的 kernel 运行期名称（`ampere_*gemm*`）与 PTX `.entry` 符号
  （mangled，如 `cgemm_largek<...>`）**不是 1:1 对应**，因此 cuBLAS 链路按
  *族*（gemm/elementwise/reduce…）匹配 PTX，而非逐 kernel 精确匹配。
- ATen 原生 fused kernel（`elementwise_kernel` 等）的 PTX 不在 cuBLAS/cuDNN/NCCL
  内（cuDNN、libtorch_cuda 为 SASS-only），可能在 Triton inductor 缓存里；
  找不到时该链路仍写出（无 `.ptx` 链接，注明 unmatched）。
- 双卡模式下 `run_dual_gpu.py` 只在 rank 0 追踪；NCCL kernel 符号跨 rank 一致，
  故 rank 0 的链路足以对应 dump 出的 PTX。

## 文件说明

| 文件 | 用途 |
|------|------|
| `run.py` | CLI 入口 — 一键跑单卡/双卡 |
| `nccl_ptx_lib/runners/run_single_gpu.py` | 单卡推理 + PTX dump |
| `nccl_ptx_lib/runners/run_dual_gpu.py` | 双卡 TP 推理 + NCCL PTX dump |
| `nccl_ptx_lib/core/env_setup.py` | 环境配置 — conda/CUDA/NCCL 路径 |
| `nccl_ptx_lib/ptx/ptx_dumper.py` | PTX 提取编排 — cuobjdump + JIT cache |
| `nccl_ptx_lib/ptx/ptx_formatter.py` | PTX 格式化 — 注释 + 分类 + 分文件 |
| `nccl_ptx_lib/chain/call_tracer.py` | 调用链路追踪 — profiler chrome trace |
| `nccl_ptx_lib/chain/chain_builder.py` | 调用链重建 — torch→aten→runtime→kernel |
| `nccl_ptx_lib/chain/runtime_knowledge.py` | 静态 runtime 层知识表 (NCCL/cuBLAS/…) |
| `nccl_ptx_lib/core/symbol_utils.py` | 符号工具 — demangle + kernel 分类 |
| `CONTEXT.md` | 共享领域术语 (Agent-friendly) |
| `docs/adr/` | 架构决策记录 |

## NCCL 本地编译（且让 torch 真正加载它）

本项目使用本地编译的 NCCL 2.21.5（`/home/zhangchen/PTX/nccl/`），用 `~/PTX/build_nccl.sh`
一键重建。两个要点：

1. **PTX 嵌入**：`NVCC_GENCODE="-gencode=arch=compute_80,code=compute_80"`
   （`compute_80` = PTX，非 `sm_80` = SASS），这样 `cuobjdump -ptx` 能提出 PTX。
2. **可加载**：NCCL device code 用 `-rdc` 编译，`nvcc -dlink` 产生的 `device_glue.o`
   会引用 `__fatbinwrap_..._cuda_device_runtime_...`，该符号由 `libcudadevrt.a` 提供。
   上游 `make src.build` 的 host link 没链 `-lcudadevrt`，导致 `.so` 无法被 `dlopen`
   （torch 加载即崩）。`build_nccl.sh` 给 `src/Makefile` 的 `LDFLAGS` 补上 `-lcudadevrt`。

### 让 torch 真正用本地 NCCL

torch 的 `libtorch_cuda.so` 带 `DT_RPATH` 指向 pip 装的 `nvidia-nccl-cu12`，且
`DT_RPATH` **优先于** `LD_LIBRARY_PATH`——光设 `LD_LIBRARY_PATH` 不够。
本项目在 `nccl_ptx_lib/core/env_setup.py` / `run.py` 里设 `LD_PRELOAD` 指向本地 `libnccl.so.2`，
`LD_PRELOAD` 优先于 `DT_RPATH`。`/proc/<pid>/maps` 可验证加载的是本地版。

### PTX 与 SASS 的区别

| | PTX | SASS |
|--|-----|------|
| 本质 | 虚拟 GPU 汇编（中间表示） | 实际 GPU 机器码 |
| 可读性 | 高（虚拟寄存器、文本指令） | 低（二进制编码、固定寄存器） |
| 编译选项 | `code=compute_80` | `code=sm_80` |
| 本地 NCCL | ✅ 已嵌入（112 个 `.entry`） | ✅ 始终包含 |

`--dump-ptx` 优先提取 PTX。如需同时输出 SASS，加 `--dump-sass`。

### 重新编译 NCCL
```bash
bash ~/PTX/build_nccl.sh
# 重建会校验 __fatbinwrap 已定义、.so 可加载
```

## 相关文档

- [CONTEXT.md](CONTEXT.md) — 共享领域术语
- [docs/adr/](docs/adr/) — 架构决策记录

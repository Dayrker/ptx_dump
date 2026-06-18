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

# 单卡推理 + dump 所有 PTX
python run.py single --dump-ptx

# 双卡推理 + dump NCCL PTX (只保留 NCCL 相关函数)
python run.py dual --dump-ptx --nccl-only

# 双卡推理 + dump + 调用链路追踪
python run.py dual --dump-ptx --nccl-only --trace-calls
```

## CLI 参考

```
python run.py <mode> [options]

Modes:
  single    单卡推理，PTX 输出到 single_ptx/
  dual      双卡 TP 推理，PTX 输出到 nccl_ptx/

Options:
  --dump-ptx          开启 PTX dump
  --nccl-only         (dual only) 只保留 NCCL 相关 kernel
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

```
run.py                   # CLI 入口 (统一参数解析 + 调度)
├── run_single_gpu.py    # 单卡推理 + tracing
├── run_dual_gpu.py      # 双卡 TP 推理 + NCCL tracing
├── env_setup.py         # 环境配置 (路径、env vars)
├── ptx_dumper.py        # PTX 提取编排 (cuobjdump + JIT)
├── ptx_formatter.py     # PTX 格式化 (注释、分类、分段)
├── call_tracer.py       # 调用链路追踪 (TorchDispatchMode + profiler)
└── symbol_utils.py      # 符号解析 (demangle + kernel 分类)
```

## PTX 可读性增强

- **符号还原**: 所有 C++ mangled name 自动 demangle
- **指令注释**: 每条 PTX 指令标注类别 (LOAD/STORE/ARITH/SYNC...)
- **Kernel 分类**: 自动分为 NCCL/cuBLAS/cuDNN/ATen 等类别
- **分文件输出**: 每个 kernel 单独一个 .ptx 文件
- **汇总表**: SUMMARY.txt 列出所有 kernel 及其元数据

## 调用链路追踪

`--trace-calls` 会记录从 PyTorch 到底层的完整调用链:

```
  [1] aten::mm
      inputs: [[4096, 4096], [4096, 4096]]
      output: [[4096, 4096]]
      └─→ CUDA kernels (1):
           ├─ ampere_sgemm_128x64_tn
           └─ ...

  [42] aten::all_reduce
      └─→ NCCL calls (1):
           ├─ ncclDevKernel_AllReduce_Sum_f16_RING_LL
      ┌─ PyTorch: dist.all_reduce()
      ├─ ATen:    c10d::all_reduce / ncclAllReduce
      ├─ NCCL:    ncclAllReduce() → Ring topology
      └─ Kernel:  ncclDevKernel_AllReduce_Sum_f16_RING_LL
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `run.py` | CLI 入口 — 一键跑单卡/双卡 |
| `run_single_gpu.py` | 单卡推理 + PTX dump |
| `run_dual_gpu.py` | 双卡 TP 推理 + NCCL PTX dump |
| `env_setup.py` | 环境配置 — conda/CUDA/NCCL 路径 |
| `ptx_dumper.py` | PTX 提取编排 — cuobjdump + JIT cache |
| `ptx_formatter.py` | PTX 格式化 — 注释 + 分类 + 分文件 |
| `call_tracer.py` | 调用链路追踪 — TorchDispatchMode + profiler |
| `symbol_utils.py` | 符号工具 — demangle + kernel 分类 |
| `CONTEXT.md` | 共享领域术语 (Agent-friendly) |
| `docs/adr/` | 架构决策记录 |

## NCCL 本地编译

本项目使用本地编译的 NCCL 2.21.5 (`/home/zhangchen/PTX/nccl/`)。

如需重新编译 (带 PTX):
```bash
cd ~/PTX/nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=compute_80"
```

> 关键: `code=compute_80` (PTX) 而非 `code=sm_80` (SASS)。

## 相关文档

- [CONTEXT.md](CONTEXT.md) — 共享领域术语
- [docs/adr/](docs/adr/) — 架构决策记录

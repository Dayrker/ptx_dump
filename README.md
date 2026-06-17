# NCCL PTX/SASS Dump — Qwen3-8B 双卡推理

## 环境

| 组件 | 版本 |
|------|------|
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| NCCL | 2.21.5 |
| GPUs | 8× A100 (sm_80) |
| Model | Qwen3-8B |

## 背景知识: NCCL 没有 PTX

NCCL 发布时 **只包含预编译的 SASS** (GPU 机器码), 不包含 PTX (中间表示)。
这是因为:

1. NCCL 的 kernel 对性能极度敏感, 必须用 SASS 级别优化
2. 发布时为每个目标架构 (sm_50 ~ sm_90) 都预编译了 SASS
3. 没有 PTX → 无法被 cuobjdump -ptx 直接提取

所以我们需要两种方法:

| 方法 | 获取内容 | 命令 |
|------|---------|------|
| **预编译 SASS** | libnccl.so 中的 GPU 机器码汇编 | `cuobjdump -sass` |
| **运行时 JIT PTX** | 如果 NCCL 有 JIT 编译的 kernel | `CUDA_JIT_CACHE_DIR` |
| **重新编译 NCCL** | 完整 PTX | 从源码编译 |

## 文件说明

```
nccl_ptx_dump/
├── run_all.sh              # 一键执行全流程
├── run_qwen3_tp.py         # Qwen3-8B 双卡 TP 推理 (触发 NCCL)
├── nccl_allreduce_test.py  # 纯 NCCL all_reduce 压测
├── extract_nccl_sass.sh    # 从 .so 提取 SASS
├── capture_jit_ptx.sh      # 运行时捕获 JIT PTX
└── README.md
```

## 快速开始

```bash
cd ~/nccl_ptx_dump
export CUDA_VISIBLE_DEVICES=0,1
bash run_all.sh
```

## 单独执行

### 1. 提取预编译 SASS (A100 sm_80)

```bash
bash extract_nccl_sass.sh
```

输出在 `sass_output/`:
- `kernel_names.txt` — 所有 NCCL kernel 函数名
- `nccl_sm80_sass.txt` — A100 的 SASS 汇编
- `nccl_res_usage.txt` — 寄存器/共享内存使用

### 2. 运行 NCCL all_reduce + 捕获 JIT PTX

```bash
bash capture_jit_ptx.sh
```

### 3. Qwen3-8B 双卡推理

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=INFO \
torchrun --nproc_per_node=2 run_qwen3_tp.py
```

## 如果确实需要 PTX (从源码编译 NCCL)

```bash
# 1. 克隆 NCCL 源码
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout v2.21.5-1

# 2. 编译, 保留 PTX
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=compute_80"

# 3. 从编译产物中提取 PTX
cuobjdump -ptx build/lib/libnccl.so.2.21.5 > nccl_from_source.ptx
```

> 注意: `code=compute_80` (PTX) 而非 `code=sm_80` (SASS) 是关键区别。

## 常用 NCCL 环境变量

| 变量 | 说明 |
|------|------|
| `NCCL_DEBUG=INFO` | 打印 NCCL 初始化信息 |
| `NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH` | 只打印指定子系统 |
| `NCCL_ALGO=Ring,Tree` | 强制使用指定算法 |
| `NCCL_PROTO=Simple,LL,LL128` | 强制使用指定协议 |
| `CUDA_JIT_CACHE_DIR=/path` | JIT cache 保存路径 |

## NCCL Kernel 分类

NCCL 的主要 kernel 家族:

| 前缀 | 说明 | 场景 |
|------|------|------|
| `ncclDevKernel_Generic` | 通用 kernel | 默认算法 |
| `ncclDevKernel_Tree*` | Tree 拓扑 | AllReduce (小规模) |
| `ncclDevKernel_Ring*` | Ring 拓扑 | AllReduce (大规模) |
| `ncclDevKernel_NVLS*` | NVLink SHARP | NVSwitch 加速 |
| `*LL128*` | Low-latency 128B | 小 tensor 优化 |
| `*LL*` | Low-latency | 小 tensor |
| `*Simple*` | Simple protocol | 大 tensor |

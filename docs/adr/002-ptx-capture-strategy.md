# ADR-002: PTX 提取策略

## 状态
已采纳（2026-06-17）

## 背景
NCCL 发布时只包含 SASS（预编译的 GPU 机器码），不包含 PTX（虚拟汇编）。要获取可读的 PTX，需要采用替代方案。

## 决策
采用多来源提取策略，按优先级依次尝试：

### 来源
1. **cuobjdump -ptx 提取 .so 中的 PTX**：仅在 NCCL 以 `code=compute_80`（嵌入 PTX）编译时有效。当前本地编译版本只有 SASS — 用户可通过重新编译来获取 PTX。
2. **CUDA JIT 缓存**：通过 `CUDA_JIT_CACHE_DIR` 在运行时捕获 JIT 编译的 kernel。部分 NCCL kernel（NVLS、collnet）是 JIT 编译的。
3. **已有的 PTX dump 文件**：复用之前编译时生成的 PTX 提取结果。
4. **SASS 提取（兜底）**：通过 `cuobjdump -sass` 提取，始终可用。

### 输出格式
- 每个 kernel 一个独立文件（demangled 名称 + 带注释的指令）
- 合并文件，包含汇总头部
- `SUMMARY.txt` 列出 kernel 分类目录

## 为什么不只用 SASS？
SASS 是架构相关的二进制汇编 — 比 PTX 难读得多。PTX 使用虚拟寄存器、可读的指令助记符，且具有可移植性。对于分析目的，PTX 是合适的抽象层级。

## 影响
- PTX 质量取决于 NCCL 的编译选项
- 如果所有 kernel 都是预编译的，JIT 缓存可能为空
- 系统始终尝试所有来源并合并结果，SASS 作为最终兜底

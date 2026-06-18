# ADR-001: CLI 架构 — 统一入口

## 状态
已采纳（2026-06-17）

## 背景
原项目由多个独立的 shell 脚本组成（`run_all.sh`、`capture_jit_ptx.sh`、`extract_nccl_sass.sh`），需要按顺序手动执行，路径硬编码，且无法通过参数控制行为。

## 决策
采用 Python CLI + 子命令的方式（`run.py single`、`run.py dual`），由入口脚本调度对应的运行模块。

### 为什么用 Python 而不是 Bash？
- `argparse` 参数解析比 shell 位置参数更健壮
- 可以直接 import 复用 PTX/trace 等模块
- 更好的错误处理和路径兼容性

### 为什么用子命令而不是 flag？
- 单卡和双卡模式的执行模型根本不同（直接运行 vs `torchrun`）
- 每种模式有各自合法的选项集（比如 `--nccl-only` 只对双卡有意义）

## 影响
- 统一入口：`python run.py <mode> [options]`
- 双卡模式内部以子进程方式调用 `torchrun`
- 环境配置集中在 `env_setup.py` 中管理

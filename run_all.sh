#!/bin/bash
# ============================================================
# run_all.sh
# 一键执行: 运行 Qwen3-8B 双卡推理 + dump NCCL SASS/PTX
# ============================================================
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Qwen3-8B Dual-GPU + NCCL PTX/SASS Dump Pipeline          ║"
echo "║  Environment: PyTorch 2.5.1 | CUDA 12.1 | 8×A100         ║"
echo "║  NCCL: 2.21.5                                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ---- 限制只用 GPU 0,1 ----
export CUDA_VISIBLE_DEVICES=0,1
echo "[ENV] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# ==== Part 1: 从 NCCL .so 中提取预编译 SASS ====
echo "=========================================="
echo "  Part 1: Extract pre-compiled SASS"
echo "=========================================="
bash "$SCRIPT_DIR/extract_nccl_sass.sh"
echo ""

# ==== Part 2: 运行 NCCL all_reduce 压测 + 捕获 JIT PTX ====
echo "=========================================="
echo "  Part 2: NCCL all_reduce + JIT capture"
echo "=========================================="
bash "$SCRIPT_DIR/capture_jit_ptx.sh"
echo ""

# ==== Part 3: Qwen3-8B 双卡推理 ====
echo "=========================================="
echo "  Part 3: Qwen3-8B TP inference"
echo "=========================================="

NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,COLL \
torchrun --nproc_per_node=2 \
    "$SCRIPT_DIR/run_qwen3_tp.py" \
    2>&1 | tee "$SCRIPT_DIR/qwen3_inference_log.txt"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  All done! Results in:                                     ║"
echo "║    sass_output/    - Pre-compiled NCCL SASS assembly       ║"
echo "║    jit_cache/      - JIT-compiled PTX (if any)             ║"
echo "║    *.txt           - Runtime logs                          ║"
echo "╚════════════════════════════════════════════════════════════╝"

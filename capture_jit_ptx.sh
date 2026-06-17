#!/bin/bash
# ============================================================
# capture_jit_ptx.sh
# 在运行时捕获 NCCL JIT 编译的 PTX
#
# NCCL 2.21+ 可能 JIT 编译部分 kernel (如 NVLS, collnet)
# CUDA driver 支持通过环境变量保存 JIT cache
# ============================================================
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
JIT_CACHE_DIR="$SCRIPT_DIR/jit_cache"

echo "============================================================"
echo "  NCCL JIT PTX Capture"
echo "============================================================"
echo ""

# 清理旧的 cache
rm -rf "$JIT_CACHE_DIR"
mkdir -p "$JIT_CACHE_DIR"

echo "  JIT cache dir: $JIT_CACHE_DIR"
echo ""

# ---- 运行 NCCL 测试, 同时捕获 JIT cache ----
echo "[Step 1] Running NCCL all_reduce with JIT cache capture..."
echo ""

CUDA_JIT_CACHE_DIR="$JIT_CACHE_DIR" \
CUDA_JIT_MAX_REGISTERS=255 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH \
torchrun --nproc_per_node=2 \
    "$SCRIPT_DIR/nccl_allreduce_test.py" \
    2>&1 | tee "$SCRIPT_DIR/nccl_runtime_log.txt"

echo ""

# ---- 分析 JIT cache ----
echo "[Step 2] Analyzing JIT cache..."
echo ""

if [ -d "$JIT_CACHE_DIR" ] && [ "$(ls -A "$JIT_CACHE_DIR" 2>/dev/null)" ]; then
    echo "  JIT cache files found:"
    find "$JIT_CACHE_DIR" -type f | while read -r f; do
        echo "    $f ($(du -h "$f" | cut -f1))"
    done
    echo ""

    # 尝试用 cuobjdump 分析 cache 中的 cubin
    echo "  Extracting PTX from JIT cache..."
    for cubin in $(find "$JIT_CACHE_DIR" -type f -name "*.cubin" -o -name "*.fatbin" 2>/dev/null); do
        echo "    Processing: $cubin"
        /usr/local/cuda-12.1/bin/cuobjdump -ptx "$cubin" 2>/dev/null || true
    done
else
    echo "  No JIT cache files found."
    echo "  (NCCL 2.21.5 may use pre-compiled SASS only, no JIT needed)"
fi

echo ""

# ---- 分析 NCCL debug log ----
echo "[Step 3] NCCL communication topology:"
echo ""
grep -E "NCCL INFO|Trees|Ring|P2P|NVLink" "$SCRIPT_DIR/nccl_runtime_log.txt" 2>/dev/null | head -40 || true

echo ""
echo "============================================================"
echo "  Runtime log:    $SCRIPT_DIR/nccl_runtime_log.txt"
echo "  JIT cache:      $JIT_CACHE_DIR"
echo "============================================================"

#!/bin/bash
# ============================================================
# extract_nccl_sass.sh
# 从 libnccl.so.2 中提取 A100 (sm_80) 的 SASS 汇编
# ============================================================
set -e

NCCL_LIB="/home/zhangchen/miniconda3/envs/torch251/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"
CUOBJDUMP="/usr/local/cuda-12.1/bin/cuobjdump"
OUT_DIR="$(dirname "$0")/sass_output"

echo "============================================================"
echo "  NCCL SASS/PTX Extraction Tool"
echo "============================================================"
echo ""
echo "  NCCL library: $NCCL_LIB"
echo "  cuobjdump:    $CUOBJDUMP"
echo "  Output dir:   $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# ---- 1. 查看 fatbin 中有哪些架构 ----
echo "[Step 1] Listing fatbin architectures..."
$CUOBJDUMP -lelf "$NCCL_LIB" > "$OUT_DIR/fatbin_list.txt" 2>&1
cat "$OUT_DIR/fatbin_list.txt"
echo ""

# ---- 2. 尝试提取 PTX (NCCL 发布时只带 SASS, 大概率没有 PTX) ----
echo "[Step 2] Attempting PTX extraction..."
$CUOBJDUMP -ptx "$NCCL_LIB" > "$OUT_DIR/nccl_ptx.txt" 2>&1 || true
if grep -q "\.version" "$OUT_DIR/nccl_ptx.txt" 2>/dev/null; then
    echo "  PTX found! Saved to: $OUT_DIR/nccl_ptx.txt"
else
    echo "  No PTX embedded (NCCL ships pre-compiled SASS only)."
    echo "  To get actual PTX, build NCCL from source (see README.md)."
fi
echo ""

# ---- 3. 提取 sm_80 (A100) SASS ----
echo "[Step 3] Extracting SASS for sm_80 (A100)..."
echo "  (This takes ~1 minute, output ~700MB...)"

$CUOBJDUMP -sass -arch sm_80 "$NCCL_LIB" > "$OUT_DIR/nccl_sm80_sass.txt" 2>&1
echo "  Full sm_80 SASS: $OUT_DIR/nccl_sm80_sass.txt ($(du -h "$OUT_DIR/nccl_sm80_sass.txt" | cut -f1))"
echo "  Total lines: $(wc -l < "$OUT_DIR/nccl_sm80_sass.txt")"
echo ""

# ---- 4. 提取 kernel 函数名 ----
echo "[Step 4] NCCL kernel catalog:"
grep "Function" "$OUT_DIR/nccl_sm80_sass.txt" | \
    sed 's/.*Function : //;s/ *$//' | sort -u > "$OUT_DIR/kernel_names.txt"
KERNEL_COUNT=$(wc -l < "$OUT_DIR/kernel_names.txt")
echo "  Total CUDA kernels: $KERNEL_COUNT"
echo "  Saved to: $OUT_DIR/kernel_names.txt"
echo ""

# 分类统计
echo "  Kernel families:"
echo "    AllReduce:  $(grep -c 'AllReduce' "$OUT_DIR/kernel_names.txt" || echo 0)"
echo "    AllGather:  $(grep -c 'AllGather' "$OUT_DIR/kernel_names.txt" || echo 0)"
echo "    ReduceScat: $(grep -c 'ReduceScatter' "$OUT_DIR/kernel_names.txt" || echo 0)"
echo "    Broadcast:  $(grep -c 'Broadcast' "$OUT_DIR/kernel_names.txt" || echo 0)"
echo "    Reduce:     $(grep -c 'ncclDev.*_Reduce_' "$OUT_DIR/kernel_names.txt" || echo 0)"
echo "    SendRecv:   $(grep -c 'SendRecv' "$OUT_DIR/kernel_names.txt" || echo 0)"
echo ""

# ---- 5. 资源使用信息 ----
echo "[Step 5] Extracting register/shared memory usage..."
$CUOBJDUMP -res-usage "$NCCL_LIB" > "$OUT_DIR/nccl_res_usage.txt" 2>&1
echo "  Saved to: $OUT_DIR/nccl_res_usage.txt"
echo ""

# ---- 6. 提取 AllReduce_Sum_f16_RING_LL 样例 ----
echo "[Step 6] Extracting AllReduce sample SASS..."
grep -n "Function.*AllReduce_Sum_f16_RING_LL[^1]" "$OUT_DIR/nccl_sm80_sass.txt" | head -2
# 提取第一个 AllReduce kernel 的前 200 行
LINE=$(grep -n "Function.*ncclDevKernel_AllReduce_Sum_f16_RING_LL" "$OUT_DIR/nccl_sm80_sass.txt" | head -1 | cut -d: -f1)
if [ -n "$LINE" ]; then
    sed -n "${LINE},$((LINE+200))p" "$OUT_DIR/nccl_sm80_sass.txt" > "$OUT_DIR/nccl_allreduce_sample.txt"
    echo "  Sample saved to: $OUT_DIR/nccl_allreduce_sample.txt"
fi
echo ""

echo "============================================================"
echo "  Done! Output files in: $OUT_DIR"
echo ""
echo "  Key files:"
echo "    kernel_names.txt           - All NCCL CUDA kernel names"
echo "    nccl_sm80_sass.txt         - Full SASS for A100 (~700MB)"
echo "    nccl_allreduce_sample.txt  - AllReduce kernel sample"
echo "    nccl_res_usage.txt         - Register/shared mem usage"
echo "============================================================"

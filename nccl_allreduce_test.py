#!/usr/bin/env python3
"""
最简版: 双卡 NCCL all_reduce 压测
保证触发 NCCL kernel, 用于配合 PTX/SASS dump

用法:
    torchrun --nproc_per_node=2 nccl_allreduce_test.py
"""

import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] Initialized on {device}")

    # 模拟 LLM 中常见的 tensor 大小
    sizes = [
        (4096,),                # hidden dim
        (4096, 4096),           # attention weight
        (1, 128, 4096),         # batch * seq * hidden
        (32, 4096, 4096),       # large batch matmul
    ]

    for size in sizes:
        tensor = torch.randn(*size, device=device, dtype=torch.float16)
        numel = tensor.numel()

        # warmup
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        N = 100
        for _ in range(N):
            dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)

        end.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end) / N  # ms per op
        bandwidth = (numel * 2 * 2) / (elapsed * 1e-3) / 1e9  # GB/s

        if rank == 0:
            shape_str = str(list(size))
            print(
                f"  all_reduce shape={shape_str:>25s}  "
                f"time={elapsed:.3f}ms  bw={bandwidth:.1f} GB/s"
            )

    if rank == 0:
        print("\nNCCL all_reduce test complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_dual_gpu.py — Qwen3-8B dual-GPU Tensor Parallel inference with NCCL PTX dump.

Designed to be launched via torchrun:
    torchrun --nproc_per_node=2 run_dual_gpu.py --dump-ptx
"""

import os
import sys
import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-8B dual-GPU TP inference")
    p.add_argument("--model-path", default="/home/model/Qwen3-8B")
    p.add_argument("--gpus", default="0,1",
                   help="指定 GPU 编号，如 '0,1' 或 '2,3' (默认: '0,1')")
    p.add_argument("--prompt", default="请用一句话解释什么是分布式训练：")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dump-ptx", action="store_true",
                   help="Dump NCCL PTX to nccl_ptx/")
    p.add_argument("--dump-sass", action="store_true",
                   help="Also dump SASS (GPU machine code) alongside PTX")
    p.add_argument("--all-kernels", action="store_true",
                   help="Dump all kernels in lib (default: only dump actually-used ones)")
    p.add_argument("--nccl-only", action="store_true",
                   help="Filter PTX to NCCL kernels only")
    p.add_argument("--trace-calls", action="store_true",
                   help="Record call chains (torch → ATen → NCCL)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    return p.parse_args()


def setup_distributed():
    """Initialize distributed environment."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def split_tensor_parallel(tensor, rank, world_size, dim=0):
    """Split tensor across GPUs."""
    chunks = torch.chunk(tensor, world_size, dim=dim)
    return chunks[rank].contiguous()


def apply_tensor_parallelism(model, rank, world_size):
    """
    Apply manual Tensor Parallelism to Qwen3-8B:
    - QKV projections: row-parallel (split output dim)
    - O projection: column-parallel (split input dim)
    - MLP gate/up: row-parallel, down: column-parallel
    """
    for name, module in model.named_modules():
        # Attention QKV
        if "self_attn" in name and hasattr(module, "q_proj"):
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(module, proj_name)
                weight = split_tensor_parallel(proj.weight, rank, world_size, dim=0)
                proj.weight = torch.nn.Parameter(weight, requires_grad=False)
                if proj.bias is not None:
                    bias = split_tensor_parallel(proj.bias, rank, world_size, dim=0)
                    proj.bias = torch.nn.Parameter(bias, requires_grad=False)

            # o_proj: column-parallel
            o_proj = module.o_proj
            weight = split_tensor_parallel(o_proj.weight, rank, world_size, dim=1)
            o_proj.weight = torch.nn.Parameter(weight, requires_grad=False)

        # MLP
        if "mlp" in name and hasattr(module, "gate_proj"):
            for proj_name in ["gate_proj", "up_proj"]:
                proj = getattr(module, proj_name)
                weight = split_tensor_parallel(proj.weight, rank, world_size, dim=0)
                proj.weight = torch.nn.Parameter(weight, requires_grad=False)

            down = module.down_proj
            weight = split_tensor_parallel(down.weight, rank, world_size, dim=1)
            down.weight = torch.nn.Parameter(weight, requires_grad=False)

    return model


def run_dual_gpu(args):
    """Run dual-GPU TP inference with NCCL tracing."""
    from env_setup import EnvConfig, setup_for_dual_gpu, setup_jit_cache
    from ptx_dumper import dump_dual_gpu_ptx
    from call_tracer import full_trace_context

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Environment
    config = setup_for_dual_gpu()
    model_path = args.model_path or config.model_path
    output_dir = args.output_dir or config.nccl_ptx_dir

    # JIT cache
    jit_dir = setup_jit_cache(config)

    if rank == 0:
        print("╔════════════════════════════════════════════════════════════╗")
        print("║  Qwen3-8B Dual-GPU Tensor Parallel Inference              ║")
        print("╚════════════════════════════════════════════════════════════╝")
        config.print_summary()
        print()

    # ─── Load model ───
    if rank == 0:
        print("[1/5] Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    if rank == 0:
        print("[2/5] Applying Tensor Parallelism...")
    model = apply_tensor_parallelism(model, rank, world_size)
    model = model.to(device)
    model.eval()

    # ─── Register AllReduce hooks ───
    call_count = [0]

    def make_allreduce_hook(layer_name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor) and out.is_cuda:
                dist.all_reduce(out, op=dist.ReduceOp.SUM)
                call_count[0] += 1
            return out
        return hook

    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "o_proj") and "self_attn" in name:
            hooks.append(module.register_forward_hook(make_allreduce_hook(f"attn:{name}")))
        if hasattr(module, "down_proj") and "mlp" in name:
            hooks.append(module.register_forward_hook(make_allreduce_hook(f"mlp:{name}")))

    if rank == 0:
        print(f"      Registered {len(hooks)} all_reduce hooks")

    # ─── Prepare input ───
    if rank == 0:
        print(f"[3/5] Preparing input...")
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    if rank == 0:
        print(f"      Prompt: {args.prompt}")
        print(f"      Tokens: {inputs['input_ids'].shape[-1]}")

    # ─── Inference (with optional tracing) ───
    if rank == 0:
        print(f"[4/5] Running inference...")

    use_tracer = (args.trace_calls or args.dump_ptx) and rank == 0
    tracer = None
    chains = None

    if use_tracer:
        with full_trace_context(trace_aten=True, trace_kernels=True) as tracer:
            # Warmup (triggers NCCL kernel compilation) — OUTSIDE the profile
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            dist.barrier()
            torch.cuda.synchronize()

            if rank == 0:
                print(f"      Warmup done. NCCL kernels compiled. Tracing the real run...")

            # Profile ONLY the real inference run
            with tracer.trace():
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                dist.barrier()
                torch.cuda.synchronize()
    else:
        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        dist.barrier()

        if rank == 0:
            print(f"      Warmup done. Full inference...")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        dist.barrier()
        torch.cuda.synchronize()

    # ─── Distributed cleanup ───
    # Keep rank-0-only trace parsing / cuobjdump work out of NCCL barriers.
    for h in hooks:
        h.remove()
    dist.barrier()
    dist.destroy_process_group()

    # ─── Output ───
    if rank == 0:
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        print(f"\n  Response: {response}")
        print(f"  NCCL all_reduce calls: {call_count[0]}")

    # ─── PTX dump (rank 0 only) ───
    if rank == 0 and args.dump_ptx:
        if use_tracer and tracer is not None:
            chains = tracer.build_chains(nccl_only=args.nccl_only)
            n_nccl = sum(1 for c in chains if c.category.startswith("NCCL"))
            print(f"      Tracing complete: {len(chains)} unique kernels chained"
                  f" ({n_nccl} NCCL)")

        print(f"\n[5/5] Dumping NCCL PTX to {output_dir}/")
        used_kernels = None if args.all_kernels else (
            {c.kernel_profiler_name for c in chains}
            if use_tracer and chains is not None else None
        )
        if used_kernels:
            print(f"      used-only 模式: 只保留运行时实际触发的 kernel")
        files = dump_dual_gpu_ptx(config, nccl_only=args.nccl_only,
                                   trace_calls=args.trace_calls,
                                   dump_sass=args.dump_sass,
                                   used_kernels=used_kernels,
                                   chains=chains)
        print(f"      Written {len(files)} files.")

        if use_tracer:
            title = (f"Dual-GPU NCCL Call Chains (torch→ATen→runtime→kernel→PTX,"
                     f" nccl_only={args.nccl_only})")
            report = tracer.write_report(output_dir, chains=chains,
                                         nccl_only=args.nccl_only, title=title)
            print(f"      Call chain report: {report}")
    elif rank == 0:
        print(f"\n[5/5] Skipping PTX dump (use --dump-ptx to enable)")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  Done! Output: {output_dir}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    run_dual_gpu(args)

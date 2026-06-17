#!/usr/bin/env python3
"""
Qwen3-8B 双卡 Tensor Parallel 推理 + NCCL 通信
触发 NCCL all_reduce 操作, 以便后续 dump 出 NCCL 的 PTX/SASS

用法:
    torchrun --nproc_per_node=2 run_qwen3_tp.py
"""

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def split_tensor_parallel(tensor, rank, world_size, dim=0):
    """将张量沿指定维度切分给各GPU"""
    chunks = torch.chunk(tensor, world_size, dim=dim)
    return chunks[rank].contiguous()


def apply_tensor_parallelism(model, rank, world_size):
    """
    对 Qwen3-8B 手动施加 Tensor Parallelism:
    - QKV projections: 按 head 切分 (dim=0, 行并行)
    - O projection: 按列切分 (dim=1, 列并行)
    - MLP gate/up: 行并行 (dim=0), down: 列并行 (dim=1)
    """
    for name, module in model.named_modules():
        # ---- Attention QKV projections (行并行: 每个GPU处理一部分 head) ----
        if "self_attn" in name and hasattr(module, "q_proj"):
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(module, proj_name)
                weight = split_tensor_parallel(proj.weight, rank, world_size, dim=0)
                proj.weight = torch.nn.Parameter(weight, requires_grad=False)
                if proj.bias is not None:
                    bias = split_tensor_parallel(proj.bias, rank, world_size, dim=0)
                    proj.bias = torch.nn.Parameter(bias, requires_grad=False)

            # o_proj: 列并行
            o_proj = module.o_proj
            weight = split_tensor_parallel(o_proj.weight, rank, world_size, dim=1)
            o_proj.weight = torch.nn.Parameter(weight, requires_grad=False)

        # ---- MLP: gate/up 行并行, down 列并行 ----
        if "mlp" in name and hasattr(module, "gate_proj"):
            for proj_name in ["gate_proj", "up_proj"]:
                proj = getattr(module, proj_name)
                weight = split_tensor_parallel(proj.weight, rank, world_size, dim=0)
                proj.weight = torch.nn.Parameter(weight, requires_grad=False)

            down = module.down_proj
            weight = split_tensor_parallel(down.weight, rank, world_size, dim=1)
            down.weight = torch.nn.Parameter(weight, requires_grad=False)

    return model


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"{'=' * 60}")
        print(f"  Qwen3-8B Tensor Parallel Inference")
        print(f"  GPUs: {world_size} | Rank: {rank} | Device: {device}")
        print(f"  NCCL Backend: {dist.get_backend()}")
        print(f"{'=' * 60}")

    # ---- 加载 tokenizer ----
    model_path = "/home/model/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---- 每个 rank 加载完整模型到 CPU, 再本地切分 ----
    if rank == 0:
        print("\n[1/4] Loading Qwen3-8B weights (CPU)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    if rank == 0:
        print("[2/4] Applying Tensor Parallelism (manual head split)...")

    # 施加 TP 切分
    model = apply_tensor_parallelism(model, rank, world_size)

    # 移到 GPU
    model = model.to(device)
    model.eval()

    if rank == 0:
        print(f"[3/4] Model on {device}, registering NCCL hooks...")

    # ---- 注册 all_reduce hook ----
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
        print(f"[4/4] Registered {len(hooks)} NCCL all_reduce hooks. Running inference...\n")

    # ---- 推理 ----
    prompt = "请用一句话解释什么是分布式训练："
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    if rank == 0:
        print(f"  Prompt: {prompt}")
        print(f"  Input shape: {inputs['input_ids'].shape}")

    # warmup: 触发 NCCL kernel JIT 编译
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    dist.barrier()

    if rank == 0:
        print(f"\n  Warmup done (NCCL kernels compiled). Full inference...\n")

    # 正式推理
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    if rank == 0:
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"\n  Response: {response}")
        print(f"\n  Total NCCL all_reduce calls: {call_count[0]}")
        print(f"\n{'=' * 60}")
        print(f"  Inference complete. NCCL has been exercised.")
        print(f"  Use extract_nccl_sass.sh to dump SASS assembly.")
        print(f"{'=' * 60}")

    # 清理
    for h in hooks:
        h.remove()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

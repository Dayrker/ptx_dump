#!/usr/bin/env python3
"""
run_single_gpu.py — Qwen3-8B single-GPU inference with optional PTX dump.

One-click: loads model, runs inference, optionally dumps PTX + call chains.
"""

import os
import sys
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-8B single-GPU inference")
    p.add_argument("--model-path", default="/home/model/Qwen3-8B")
    p.add_argument("--prompt", default="请用一句话解释什么是深度学习：")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dump-ptx", action="store_true",
                   help="Dump all CUDA PTX to single_ptx/")
    p.add_argument("--trace-calls", action="store_true",
                   help="Record call chains (torch → ATen → CUDA)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    return p.parse_args()


def run_single_gpu(args):
    """Run single-GPU inference with optional tracing."""
    from env_setup import EnvConfig, setup_for_single_gpu, setup_jit_cache
    from ptx_dumper import dump_single_gpu_ptx
    from call_tracer import full_trace_context

    # Environment
    config = setup_for_single_gpu()
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)

    model_path = args.model_path or config.model_path
    output_dir = args.output_dir or config.single_ptx_dir

    # JIT cache (for PTX capture)
    jit_dir = setup_jit_cache(config)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Qwen3-8B Single-GPU Inference                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    config.print_summary()
    print()

    # ─── Load model ───
    print("[1/4] Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("cuda:0")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Model loaded: {n_params / 1e9:.1f}B params on cuda:0")

    # ─── Prepare input ───
    print(f"[2/4] Preparing input...")
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    print(f"      Prompt: {args.prompt}")
    print(f"      Tokens: {inputs['input_ids'].shape[-1]}")

    # ─── Inference (with optional tracing) ───
    print(f"[3/4] Running inference...")
    use_tracer = args.trace_calls or args.dump_ptx

    if use_tracer:
        with full_trace_context(trace_aten=True, trace_kernels=True) as tracer:
            # Warmup
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            torch.cuda.synchronize()

            # Real inference
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()

        print(f"      Tracing complete.")
    else:
        with torch.no_grad():
            # Warmup
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            torch.cuda.synchronize()

            # Real inference
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()

    # ─── Output ───
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    print(f"\n  Response: {response}")

    # ─── PTX dump ───
    if args.dump_ptx:
        print(f"\n[4/4] Dumping PTX to {output_dir}/")
        files = dump_single_gpu_ptx(config, trace_calls=args.trace_calls)
        print(f"      Written {len(files)} files.")
    else:
        print(f"\n[4/4] Skipping PTX dump (use --dump-ptx to enable)")

    # ─── Call chain report ───
    if use_tracer and (args.trace_calls or args.dump_ptx):
        report = tracer.write_report(output_dir, title="Single-GPU Call Chains")
        print(f"      Call chain report: {report}")

    print(f"\n{'=' * 60}")
    print(f"  Done! Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()
    run_single_gpu(args)

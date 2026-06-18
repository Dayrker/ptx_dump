#!/usr/bin/env python3
"""
run.py — Unified CLI for NCCL PTX dump project.

Usage:
    # Single-GPU Qwen3-8B + PTX dump
    python run.py single --dump-ptx

    # Dual-GPU Qwen3-8B + NCCL-only PTX dump
    python run.py dual --dump-ptx --nccl-only

    # Dual-GPU with call chain tracing
    python run.py dual --dump-ptx --nccl-only --trace-calls

    # Custom model path
    python run.py single --model-path /path/to/model --dump-ptx
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="NCCL PTX Dump — Qwen3-8B inference with PTX extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py single --dump-ptx              # Single GPU, dump all PTX
  python run.py dual --dump-ptx                # Dual GPU, dump all PTX
  python run.py dual --dump-ptx --nccl-only    # Dual GPU, NCCL kernels only
  python run.py dual --dump-ptx --trace-calls  # Dual GPU, dump + trace calls
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="GPU mode")

    # ─── Common arguments ───
    def add_common_args(p):
        p.add_argument("--model-path", default="/home/model/Qwen3-8B",
                       help="Path to Qwen3-8B model")
        p.add_argument("--gpus", default=None,
                       help="指定 GPU 编号，如 '0,1' 或 '2,3' (默认: 单卡='0', 双卡='0,1')")
        p.add_argument("--prompt", default=None,
                       help="Custom prompt for inference")
        p.add_argument("--max-new-tokens", type=int, default=None,
                       help="Max tokens to generate")
        p.add_argument("--dump-ptx", action="store_true",
                       help="Dump PTX to output directory")
        p.add_argument("--dump-sass", action="store_true",
                       help="Also dump SASS (GPU machine code) alongside PTX")
        p.add_argument("--all-kernels", action="store_true",
                       help="Dump all kernels in lib (default: only dump actually-used ones)")
        p.add_argument("--trace-calls", action="store_true",
                       help="Record torch → ATen → CUDA call chains")
        p.add_argument("--output-dir", default=None,
                       help="Override output directory")

    # ─── Single-GPU mode ───
    single_p = subparsers.add_parser("single", help="Single-GPU inference")
    add_common_args(single_p)
    if single_p.get_default("prompt") is None:
        single_p.set_defaults(prompt="请用一句话解释什么是深度学习：")
        single_p.set_defaults(max_new_tokens=64)

    # ─── Dual-GPU mode ───
    dual_p = subparsers.add_parser("dual", help="Dual-GPU Tensor Parallel inference")
    add_common_args(dual_p)
    dual_p.add_argument("--nccl-only", action="store_true",
                        help="Only dump NCCL-related PTX kernels")
    if dual_p.get_default("prompt") is None:
        dual_p.set_defaults(prompt="请用一句话解释什么是分布式训练：")
        dual_p.set_defaults(max_new_tokens=128)

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Build command for the appropriate runner
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "single":
        gpus = args.gpus or "0"
        cmd = [
            sys.executable,
            os.path.join(script_dir, "run_single_gpu.py"),
            "--model-path", args.model_path,
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
            "--gpus", gpus,
        ]
        if args.dump_ptx:
            cmd.append("--dump-ptx")
        if args.dump_sass:
            cmd.append("--dump-sass")
        if args.all_kernels:
            cmd.append("--all-kernels")
        if args.trace_calls:
            cmd.append("--trace-calls")
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])

    elif args.mode == "dual":
        gpus = args.gpus or "0,1"
        # Dual GPU needs torchrun
        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            os.path.join(script_dir, "run_dual_gpu.py"),
            "--model-path", args.model_path,
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
            "--gpus", gpus,
        ]
        if args.dump_ptx:
            cmd.append("--dump-ptx")
        if args.dump_sass:
            cmd.append("--dump-sass")
        if args.all_kernels:
            cmd.append("--all-kernels")
        if args.nccl_only:
            cmd.append("--nccl-only")
        if args.trace_calls:
            cmd.append("--trace-calls")
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])

        # Set environment for torchrun
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"

        # Add local NCCL to LD_LIBRARY_PATH
        nccl_lib = "/home/zhangchen/PTX/nccl/build/lib"
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{nccl_lib}:{ld_path}" if ld_path else nccl_lib

        print(f"  Launching torchrun with CUDA_VISIBLE_DEVICES={gpus}")
        print(f"  NCCL lib: {nccl_lib}")
        print()

        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)

    # Execute
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

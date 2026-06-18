#!/usr/bin/env python3
"""
env_setup.py — Environment configuration for NCCL PTX dump project.

Resolves all paths, validates dependencies, and sets environment variables
for CUDA, NCCL, and conda.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Validated environment configuration."""

    # CUDA
    cuda_home: str = "/usr/local/cuda-12.1"
    cuobjdump: str = "/usr/local/cuda-12.1/bin/cuobjdump"
    cppfilt: str = "/usr/bin/c++filt"

    # NCCL (local build)
    nccl_lib_dir: str = "/home/zhangchen/PTX/nccl/build/lib"
    nccl_include_dir: str = "/home/zhangchen/PTX/nccl/build/include"
    nccl_lib: str = "/home/zhangchen/PTX/nccl/build/lib/libnccl.so.2.21.5"

    # Conda
    conda_env: str = "torch251"

    # Model
    model_path: str = "/home/model/Qwen3-8B"

    # Project
    project_dir: str = ""
    single_ptx_dir: str = ""
    nccl_ptx_dir: str = ""

    # GPU
    visible_devices: str = ""
    num_gpus: int = 1

    def __post_init__(self):
        if not self.project_dir:
            self.project_dir = str(Path(__file__).parent)
        if not self.single_ptx_dir:
            self.single_ptx_dir = os.path.join(self.project_dir, "single_ptx")
        if not self.nccl_ptx_dir:
            self.nccl_ptx_dir = os.path.join(self.project_dir, "nccl_ptx")

    def validate(self) -> list:
        """Validate environment. Returns list of error messages."""
        errors = []

        if not os.path.isfile(self.cuobjdump):
            errors.append(f"cuobjdump not found: {self.cuobjdump}")

        if not os.path.isfile(self.nccl_lib):
            errors.append(f"Local NCCL lib not found: {self.nccl_lib}")

        if not os.path.isdir(self.nccl_include_dir):
            errors.append(f"Local NCCL include not found: {self.nccl_include_dir}")

        if not os.path.isdir(self.model_path):
            errors.append(f"Model not found: {self.model_path}")

        # Check torch
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA not available via torch")
        except ImportError:
            errors.append(
                "PyTorch not installed. Run: conda activate torch251"
            )

        return errors

    def setup_env(self):
        """Set environment variables for NCCL + CUDA."""
        # Prepend local NCCL to LD_LIBRARY_PATH
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{self.nccl_lib_dir}:{ld_path}" if ld_path else self.nccl_lib_dir

        # CUDA paths
        os.environ["CUDA_HOME"] = self.cuda_home
        os.environ["PATH"] = f"{self.cuda_home}/bin:{os.environ.get('PATH', '')}"

        # NCCL visibility
        if self.visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_devices

    def print_summary(self):
        """Print environment summary."""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║  NCCL PTX Dump — Environment                              ║")
        print("╠════════════════════════════════════════════════════════════╣")
        print(f"║  CUDA:        {self.cuda_home:<45s} ║")
        print(f"║  cuobjdump:   {self.cuobjdump:<45s} ║")
        print(f"║  NCCL lib:    {self.nccl_lib:<45s} ║")
        print(f"║  Model:       {self.model_path:<45s} ║")
        print(f"║  GPUs:        {self.visible_devices:<45s} ║")
        print(f"║  LD_LIBRARY:  {self.nccl_lib_dir:<45s} ║")
        print("╚════════════════════════════════════════════════════════════╝")


def setup_for_single_gpu(config: EnvConfig = None) -> EnvConfig:
    """Configure for single-GPU mode."""
    if config is None:
        config = EnvConfig()
    config.visible_devices = "0"
    config.num_gpus = 1
    config.setup_env()
    return config


def setup_for_dual_gpu(config: EnvConfig = None) -> EnvConfig:
    """Configure for dual-GPU mode."""
    if config is None:
        config = EnvConfig()
    config.visible_devices = "0,1"
    config.num_gpus = 2
    config.setup_env()
    return config


def setup_jit_cache(config: EnvConfig, cache_dir: str = None) -> str:
    """Configure JIT cache directory for PTX capture."""
    if cache_dir is None:
        cache_dir = os.path.join(config.project_dir, ".jit_cache")

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["CUDA_JIT_CACHE_DIR"] = cache_dir
    os.environ["CUDA_JIT_MAX_REGISTERS"] = "255"

    return cache_dir

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

    # Other CUDA libs whose kernels run during inference (single-GPU mode).
    # cuBLAS ships PTX embedded in its .so; the others may be SASS-only.
    cublas_lib: str = ""
    cudnn_lib: str = ""
    cufft_lib: str = ""
    curand_lib: str = ""
    # torch's Triton/inductor JIT cache (fused elementwise/reduction kernels)
    torchinductor_cache: str = ""
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
            # repo root = 3 dirs up from nccl_ptx_lib/core/env_setup.py
            # (core -> nccl_ptx_lib -> repo root); outputs land at repo root.
            self.project_dir = str(Path(__file__).parent.parent.parent)
        if not self.single_ptx_dir:
            self.single_ptx_dir = os.path.join(self.project_dir, "single_ptx")
        if not self.nccl_ptx_dir:
            self.nccl_ptx_dir = os.path.join(self.project_dir, "nccl_ptx")
        # Auto-detect torch's bundled CUDA libs + Triton cache (single-GPU mode
        # dumps PTX from these, since single-GPU inference uses cuBLAS/cuDNN/ATen
        # kernels, NOT NCCL).
        self._autodetect_cuda_libs()

    def _autodetect_cuda_libs(self):
        """Find cuBLAS/cuDNN/cuFFT/cuRAND and the Triton inductor cache."""
        try:
            import torch as _t
            torch_dir = os.path.dirname(_t.__file__)
            site = os.path.dirname(torch_dir)
        except Exception:
            return
        nv = os.path.join(site, "nvidia")
        candidates = {
            "cublas_lib": (os.path.join(nv, "cublas", "lib"), "libcublas.so"),
            "cudnn_lib": (os.path.join(nv, "cudnn", "lib"), "libcudnn.so"),
            "cufft_lib": (os.path.join(nv, "cufft", "lib"), "libcufft.so"),
            "curand_lib": (os.path.join(nv, "curand", "lib"), "libcurand.so"),
        }
        for attr, (d, prefix) in candidates.items():
            if getattr(self, attr):
                continue
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith(prefix):
                        setattr(self, attr, os.path.join(d, f))
                        break
        # Triton/inductor cache (runtime JIT .ptx for fused kernels)
        if not self.torchinductor_cache:
            import glob as _g
            for cand in _g.glob("/tmp/torchinductor_*"):
                self.torchinductor_cache = cand
                break

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
        """Set environment variables for NCCL + CUDA.

        torch's libtorch_cuda.so carries a DT_RPATH that points at the pip
        nvidia-nccl, and DT_RPATH takes precedence over LD_LIBRARY_PATH — so
        setting LD_LIBRARY_PATH alone is NOT enough to make torch load the
        locally-built NCCL. We use LD_PRELOAD to force it. This requires the
        local NCCL to be loadable (it links libcudadevrt; see build_nccl.sh)."""
        nccl_so = os.path.join(self.nccl_lib_dir, "libnccl.so.2")

        # Prepend local NCCL to LD_LIBRARY_PATH (covers cuobjdump / other tools)
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{self.nccl_lib_dir}:{ld_path}" if ld_path else self.nccl_lib_dir

        # Force torch to load the LOCAL NCCL at runtime (beats DT_RPATH).
        # Only set if the local .so is actually loadable; otherwise leave it to
        # the pip NCCL so the process still runs.
        if os.path.exists(nccl_so):
            existing = os.environ.get("LD_PRELOAD", "")
            if nccl_so not in existing:
                os.environ["LD_PRELOAD"] = (
                    f"{nccl_so}:{existing}" if existing else nccl_so
                )

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

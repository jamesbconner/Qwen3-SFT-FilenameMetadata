"""Verify CUDA + attention backends (flash-attn / xformers / SDPA) + Unsloth."""
import importlib
import sys
import warnings
import logging

# Suppress the torch.distributed warning about redirects on Windows/MacOS
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch

def try_import(name: str):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "unknown")
        return True, ver, m
    except Exception as e:
        return False, repr(e), None

print("=" * 60)
print("Environment Information")
print("=" * 60)
print(f"Python Version: {sys.version.split()[0]} ({sys.platform})")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version (torch): {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    t = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    _ = t @ t
    print("[OK] GPU matmul works (fp16)")

print("\n" + "=" * 60)
print("Attention Backends (what your training will actually use)")
print("=" * 60)

# PyTorch SDPA toggles
if hasattr(torch.backends, "cuda"):
    print(f"SDPA flash_sdp_enabled:          {getattr(torch.backends.cuda, 'flash_sdp_enabled', lambda: 'n/a')()}")
    print(f"SDPA mem_efficient_sdp_enabled:  {getattr(torch.backends.cuda, 'mem_efficient_sdp_enabled', lambda: 'n/a')()}")
    print(f"SDPA math_sdp_enabled:           {getattr(torch.backends.cuda, 'math_sdp_enabled', lambda: 'n/a')()}")

# Optional libs
ok, info, xformers_mod = try_import("xformers")
print(f"xformers importable:             {ok} ({info})")

# Check Flash Attention availability
if ok and xformers_mod:
    try:
        from xformers.ops import memory_efficient_attention
        # Test if FA works (this will use the best available backend)
        if torch.cuda.is_available():
            q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
            _ = memory_efficient_attention(q, k, v)
            print(f"Flash Attention (xformers):      True")
        else:
            print(f"Flash Attention (xformers):      N/A (CUDA not available)")
    except Exception as e:
        print(f"Flash Attention (xformers):      False ({e})")
else:
    print(f"Flash Attention (xformers):      N/A (xformers not available)")

# Check standalone flash-attn package
ok_flash, info_flash, _ = try_import("flash_attn")
if ok_flash:
    print(f"flash-attn (standalone):         True ({info_flash})")
else:
    print(f"flash-attn (standalone):         False")

# Check Triton availability
try:
    import triton
    print(f"Triton importable:               True ({triton.__version__})")
except ImportError:
    # Check through xformers if available
    if ok and xformers_mod:
        try:
            from xformers import is_triton_available
            triton_avail = is_triton_available()
            print(f"Triton (via xformers):            {triton_avail}")
        except:
            print(f"Triton importable:              False")
    else:
        print(f"Triton importable:              False")
except Exception as e:
    print(f"Triton importable:              False ({e})")

print("\n" + "=" * 60)
print("Unsloth Verification")
print("=" * 60)

ok, info, unsloth_mod = try_import("unsloth")
if not ok:
    print(f"[ERROR] Unsloth import failed: {info}")
else:
    print(f"[OK] Unsloth imported: {info}")
    # FastLanguageModel is a class in the unsloth module, not a submodule
    try:
        from unsloth import FastLanguageModel
        print(f"FastLanguageModel importable:    True (class: {type(FastLanguageModel).__name__})")
    except Exception as e:
        print(f"FastLanguageModel importable:    False ({e})")

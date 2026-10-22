"""Compile RMSNorm after ZeRO gathers with a shared checkpoint Q/K forward."""

from types import MethodType
from collections import Counter
import atexit
import math
import os
from pathlib import Path
import shutil
import tempfile

# Each process owns fresh caches, avoiding concurrent helper-library replacement
# in the shared Triton home cache. Explicit user cache settings take precedence.
_cache_root = Path(tempfile.mkdtemp(prefix="minimax-h3-compile-"))
(_cache_root / ".owner").write_text(str(os.getpid()))
os.environ.setdefault("TRITON_CACHE_DIR", str(_cache_root / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_root / "inductor"))


def _cleanup():
    marker = _cache_root / ".owner"
    if marker.is_file() and marker.read_text() == str(os.getpid()):
        removed = sum(p.stat().st_size for p in _cache_root.rglob("*") if p.is_file())
        shutil.rmtree(_cache_root)
        print(f"RMSNorm compile cache cleanup: bytes={removed} files_left=0", flush=True)


atexit.register(_cleanup)

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

_calls = 0
_eager_qk_modules = 0
_checkpoint_qk_modules = 0
_qk_forward_calls = _qk_backward_calls = 0
_qk_fallbacks = Counter()


def _rms_norm(value, weight, eps):
    return F.rms_norm(value, (value.shape[-1],), weight, eps)


_compiled = torch.compile(
    _rms_norm, fullgraph=True, dynamic=True,
    options={"emulate_precision_casts": True},
)


class _CheckpointQK(torch.autograd.Function):
    """Share the forward; reuse the existing first-order RMSNorm derivative."""

    @staticmethod
    def forward(ctx, value, weight, eps):
        global _calls, _qk_forward_calls
        ctx.save_for_backward(value.detach(), weight.detach())
        ctx.eps = eps
        _calls += 1
        _qk_forward_calls += 1
        return _compiled(value.detach(), weight.detach(), eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        global _calls, _qk_backward_calls
        if not ctx.needs_input_grad[0]:
            return None, None, None
        value, weight = ctx.saved_tensors
        # Recompute only this pure norm, not its module/ZeRO hooks. Autograd
        # supplies the existing compiled FP32-intermediate native derivative.
        with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
            leaf = value.detach().requires_grad_(True)
            _calls += 1
            _qk_backward_calls += 1
            output = _compiled(leaf, weight, ctx.eps)
            (gradient,) = torch.autograd.grad(output, leaf, grad, create_graph=False, retain_graph=False)
        return gradient, None, None


def _checkpoint_qk_forward(module, value):
    weight = module.weight
    reason = None
    if weight is None or weight.requires_grad:
        reason = "missing_or_trainable_weight"
    elif (value.device.type != "cuda" or weight.device != value.device
          or value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16):
        reason = "device_or_dtype"
    elif (value.ndim == 0 or value.shape[-1] != module.normalized_shape[0]
          or value.stride(-1) != 1 or weight.ndim != 1 or weight.stride(0) != 1):
        reason = "shape_or_stride"
    elif module.eps is None or not math.isfinite(module.eps) or module.eps <= 0:
        reason = "epsilon"
    elif torch.is_autocast_enabled("cuda"):
        reason = "autocast"
    if reason is not None:
        _qk_fallbacks[reason] += 1
        return torch.nn.RMSNorm.forward(module, value)
    # Pass the original activation to apply: its returned dx connects to LoRA.
    return _CheckpointQK.apply(value, weight, module.eps)


def _forward(module, value):
    global _calls
    _calls += 1
    return _compiled(value, module.weight, module.eps)


def install(model):
    global _eager_qk_modules, _checkpoint_qk_modules
    torch._logging.set_logs(recompiles=True)
    qk_norms = {
        id(norm)
        for module in model.modules()
        for norm in (getattr(module, "norm_q", None), getattr(module, "norm_k", None))
        if type(norm) is torch.nn.RMSNorm and len(norm.normalized_shape) == 1
    }
    count = 0
    _eager_qk_modules = 0
    _checkpoint_qk_modules = 0
    for module in model.modules():
        if type(module) is torch.nn.RMSNorm and len(module.normalized_shape) == 1:
            if id(module) in qk_norms:
                # Cached O/LSE need identical Q/K in both checkpoint passes.
                # Keep user overrides; replace only native or our own methods.
                if getattr(module.forward, "__func__", None) in (
                    torch.nn.RMSNorm.forward, _forward, _checkpoint_qk_forward,
                ):
                    module.forward = MethodType(_checkpoint_qk_forward, module)
                    _checkpoint_qk_modules += 1
                    count += 1
                else:
                    _eager_qk_modules += 1
                continue
            module.forward = MethodType(_forward, module)
            count += 1
    return count


def report():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    return {"calls": _calls, "eager_qk_modules": _eager_qk_modules,
            "checkpoint_qk_modules": _checkpoint_qk_modules,
            "qk_forward_calls": _qk_forward_calls, "qk_backward_calls": _qk_backward_calls,
            "qk_fallbacks": dict(_qk_fallbacks),
            "graphs": dict(counters["stats"]),
            "inductor": dict(counters["inductor"]),
            "generated_kernels": metrics.generated_kernel_count}

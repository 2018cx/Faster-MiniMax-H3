"""Compile RMSNorm after ZeRO gathers, keeping checkpointed Q/K norms native."""

from types import MethodType
import atexit
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
from torch.nn import functional as F

_calls = 0
_eager_qk_modules = 0


def _rms_norm(value, weight, eps):
    return F.rms_norm(value, (value.shape[-1],), weight, eps)


_compiled = torch.compile(
    _rms_norm, fullgraph=True, dynamic=True,
    options={"emulate_precision_casts": True},
)


def _forward(module, value):
    global _calls
    _calls += 1
    return _compiled(value, module.weight, module.eps)


def install(model):
    global _eager_qk_modules
    torch._logging.set_logs(recompiles=True)
    qk_norms = {
        id(norm)
        for module in model.modules()
        for norm in (getattr(module, "norm_q", None), getattr(module, "norm_k", None))
        if type(norm) is torch.nn.RMSNorm and len(norm.normalized_shape) == 1
    }
    count = 0
    _eager_qk_modules = 0
    for module in model.modules():
        if type(module) is torch.nn.RMSNorm and len(module.normalized_shape) == 1:
            if id(module) in qk_norms:
                # Cached attention's O/LSE must match replayed Q/K. Inductor
                # can select different RMS reductions in no-grad and grad
                # forwards, so keep these norms on the same native path.
                if getattr(module.forward, "__func__", None) is _forward:
                    del module.forward
                _eager_qk_modules += 1
                continue
            module.forward = MethodType(_forward, module)
            count += 1
    return count


def report():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    return {"calls": _calls, "eager_qk_modules": _eager_qk_modules,
            "graphs": dict(counters["stats"]),
            "inductor": dict(counters["inductor"]),
            "generated_kernels": metrics.generated_kernel_count}

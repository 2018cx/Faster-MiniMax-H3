"""Fused native RoPE arithmetic and cuDNN layout, with eager rounding."""

from collections import Counter

import compiled_norm as _compiler_cache  # Shared process-owned compiler caches.
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

_calls = 0
_variants = set()
_fallbacks = Counter()
_checked = set()
_disabled = False


@triton.jit
def _rotate(X, C, S, O, N: tl.constexpr, LENGTH: tl.constexpr,
            HEADS: tl.constexpr, WIDTH: tl.constexpr, ROTARY: tl.constexpr,
            XB: tl.constexpr, XT: tl.constexpr, XH: tl.constexpr, XD: tl.constexpr,
            CT: tl.constexpr, CD: tl.constexpr, ST: tl.constexpr, SD: tl.constexpr,
            BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    channel = index % WIDTH
    if BACKWARD:
        head = (index // WIDTH) % HEADS
        token = (index // (WIDTH * HEADS)) % LENGTH
    else:
        token = (index // WIDTH) % LENGTH
        head = (index // (WIDTH * LENGTH)) % HEADS
    batch = index // (WIDTH * LENGTH * HEADS)
    offset = batch * XB + token * XT + head * XH + channel * XD
    first_half = channel < ROTARY // 2
    partner = tl.where(first_half, channel + ROTARY // 2, channel - ROTARY // 2)
    active = valid & (channel < ROTARY)
    dtype = O.dtype.element_ty
    value = tl.load(X + offset, valid, other=0).to(tl.float32)
    other = tl.load(X + offset + (partner - channel) * XD, active, other=0).to(tl.float32)
    cosine = tl.load(C + token * CT + channel * CD, active, other=0).to(dtype).to(tl.float32)
    sine_channel = partner if BACKWARD else channel
    sine = tl.load(S + token * ST + sine_channel * SD, active, other=0).to(dtype).to(tl.float32)
    first_product = (value * cosine).to(dtype).to(tl.float32)
    if BACKWARD:
        second_product = (other * sine).to(dtype).to(tl.float32)
        second_product = tl.where(first_half, second_product, -second_product)
    else:
        other = tl.where(first_half, -other, other)
        second_product = (other * sine).to(dtype).to(tl.float32)
    result = (first_product + second_product).to(dtype)
    result = tl.where(channel < ROTARY, result, value.to(dtype))
    tl.store(O + index, result, valid)


@triton_op("minimax_h3::rotary_layout", mutates_args={})
def _forward(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    batch, length, heads, width = value.shape
    output = torch.empty((batch, heads, length, width), device=value.device, dtype=value.dtype)
    wrap_triton(_rotate)[(triton.cdiv(value.numel(), 1024),)](
        value, cos, sin, output, value.numel(), length, heads, width, cos.shape[-1],
        *value.stride(), *cos.stride(), *sin.stride(), False, 1024,
        num_warps=4, enable_fp_fusion=False)
    return output


@triton_op("minimax_h3::rotary_layout_backward", mutates_args={})
def _backward(grad: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    batch, heads, length, width = grad.shape
    output = torch.empty((batch, length, heads, width), device=grad.device, dtype=grad.dtype)
    gb, gh, gt, gd = grad.stride()
    wrap_triton(_rotate)[(triton.cdiv(grad.numel(), 1024),)](
        grad, cos, sin, output, grad.numel(), length, heads, width, cos.shape[-1],
        gb, gt, gh, gd, *cos.stride(), *sin.stride(), True, 1024,
        num_warps=4, enable_fp_fusion=False)
    return output


def _setup(ctx, inputs, output):
    _, cos, sin = inputs
    ctx.save_for_backward(cos, sin)


def _autograd(ctx, grad):
    return _backward(grad, *ctx.saved_tensors), None, None


_forward.register_autograd(_autograd, setup_context=_setup)


def _native(value, cos, sin):
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    return _apply_rotary_emb(value, cos, sin).transpose(1, 2).contiguous()


def rotary_head_major(value, cos, sin):
    global _calls, _disabled
    reason = None
    if _disabled:
        reason = "first_call_mismatch"
    elif not value.is_cuda or value.dtype not in (torch.bfloat16, torch.float32):
        reason = "device_or_dtype"
    elif cos.device != value.device or sin.device != value.device:
        reason = "table_device"
    elif cos.dtype not in (torch.bfloat16, torch.float32) or sin.dtype not in (torch.bfloat16, torch.float32):
        reason = "table_dtype"
    elif cos.requires_grad or sin.requires_grad:
        reason = "trainable_rotary_tables"
    elif value.numel() == 0:
        reason = "empty_input"
    elif (value.ndim != 4 or cos.ndim != 2 or sin.shape != cos.shape
          or cos.shape[0] != value.shape[1] or cos.shape[-1] > value.shape[-1]
          or cos.shape[-1] % 2):
        reason = "rotary_shape"
    if reason:
        _fallbacks[reason] += 1
        return _native(value, cos, sin)
    _calls += 1
    key = (tuple(value.shape), value.stride(), value.dtype, value.device,
           cos.shape, cos.stride(), cos.dtype, sin.stride(), sin.dtype)
    if key not in _variants:
        _variants.add(key)
        print(f"RoPE kernel variant: {len(_variants)} shape={tuple(value.shape)} dtype={value.dtype}", flush=True)
    output = _forward(value, cos, sin)
    if key not in _checked:
        reference = _native(value, cos, sin)
        if not torch.equal(output, reference):
            _disabled = True
            _fallbacks["first_call_mismatch"] += 1
            print("RoPE first-call equality FAILED; fused path disabled", flush=True)
            return reference
        _checked.add(key)
        print("RoPE first-call equality PASS", flush=True)
    return output


def report():
    return {"calls": _calls, "variants": len(_variants), "verified_variants": len(_checked),
            "disabled": _disabled, "fallbacks": dict(_fallbacks)}

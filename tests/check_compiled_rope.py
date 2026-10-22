"""Numerical checks only; benchmark performance through the frozen bench."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
from compiled_rope import rotary_head_major


def check(length, heads, width, rotary_width, dtype, strided=False):
    value = torch.randn(2 if strided else 1, length, heads, width * (2 if strided else 1), device="cuda", dtype=dtype,
                        requires_grad=True)
    if strided:
        value = value[..., ::2]
    angles = torch.randn(length, rotary_width, device="cuda", dtype=torch.float32)
    cos, sin = angles.cos(), angles.sin()
    expected = _apply_rotary_emb(value, cos, sin).transpose(1, 2).contiguous()
    actual = rotary_head_major(value, cos, sin)
    grad = torch.randn_like(expected)
    if strided:
        grad = torch.randn(*expected.shape, 2, device="cuda", dtype=dtype)[..., 0]
    expected_grad = torch.autograd.grad(expected, value, grad)[0]
    actual_grad = torch.autograd.grad(actual, value, grad)[0]
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    with torch.no_grad():
        first = rotary_head_major(value, cos, sin)
    torch.testing.assert_close(first, expected.detach(), rtol=0, atol=0)
    print(f"bitwise rotary/layout output and input-gradient PASS: "
          f"S={length} H={heads} D={width} rotary={rotary_width} {dtype}", flush=True)


if __name__ == "__main__":
    torch.manual_seed(1234)
    check(129, 4, 128, 64, torch.bfloat16)
    check(256, 4, 128, 128, torch.bfloat16)
    check(129, 4, 128, 128, torch.float32)
    check(33750, 56, 128, 128, torch.bfloat16)
    check(33750, 56, 128, 96, torch.bfloat16)
    check(129, 4, 128, 64, torch.bfloat16, strided=True)

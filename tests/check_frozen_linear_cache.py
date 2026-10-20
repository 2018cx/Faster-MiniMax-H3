"""Check byte-exact outputs/gradients and asynchronous checkpoint prefetch.

This is numerical validation only. Use bench.py for every performance result.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import _get_current_dispatch_mode

from attention_checkpoint import _Cache, _active
from frozen_linear_cache import install, _Arena


def invoke(module, input, cache, replay):
    token = _active.set((cache, replay))
    try:
        return module(input)
    finally:
        _active.reset(token)


def check(shape, cpu, dtype=torch.bfloat16, bias=False, entry_gib=3):
    rows, inputs, outputs = shape
    module = torch.nn.Linear(inputs, outputs, bias=bias, device="cuda", dtype=dtype)
    module.requires_grad_(False)
    owner = install(module, gpu_gib=0 if cpu else 3, cpu_gib=2 if cpu else 0,
                    max_entry_gib=entry_gib)
    value = torch.randn(1, rows, inputs, dtype=dtype, device="cuda", requires_grad=True)
    expected = F.linear(value, module.weight, module.bias)
    gradient = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, value, gradient)[0]
    cache = _Cache(SimpleNamespace(retained=0))
    with torch.no_grad():
        first = invoke(module, value, cache, False)
    replay_input = value.detach().requires_grad_()
    actual = invoke(module, replay_input, cache, True)
    cache.clear()  # Real DeepSpeed replay clears records before local backward.
    actual_grad = torch.autograd.grad(actual, replay_input, gradient)[0]
    assert torch.equal(first, expected)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad, expected_grad)
    assert owner.gpu_live == owner.cpu_live == 0
    if entry_gib:
        assert owner.counts["replayed_ops"] == 1
    else:
        assert owner.counts["oversized_ops"] == owner.counts["recomputed_calls"] == 1
    assert _get_current_dispatch_mode() is None
    print(f"bitwise Linear output/input-gradient PASS: {shape}, cpu={cpu}, dtype={dtype}", flush=True)


def check_prefetch():
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16),
        torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16))
    model.requires_grad_(False)
    owner = install(model, gpu_gib=0, cpu_gib=1 / 64)
    value = torch.randn(1, 257, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = F.linear(F.linear(value, model[0].weight), model[1].weight)
    gradient = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, value, gradient)[0]
    caches = [_Cache(SimpleNamespace(retained=0)) for _ in range(2)]
    with torch.no_grad():
        middle = invoke(model[0], value, caches[0], False)
        invoke(model[1], middle, caches[1], False)
    middle_input = middle.detach().requires_grad_()
    actual = invoke(model[1], middle_input, caches[1], True)
    caches[1].clear()
    middle_grad = torch.autograd.grad(actual, middle_input, gradient)[0]
    replay_input = value.detach().requires_grad_()
    middle = invoke(model[0], replay_input, caches[0], True)
    caches[0].clear()
    actual_grad = torch.autograd.grad(middle, replay_input, middle_grad)[0]
    assert torch.equal(actual, expected) and torch.equal(actual_grad, expected_grad)
    assert owner.counts["prefetched_blocks"] >= 1
    assert owner.gpu_live == owner.cpu_live == 0
    print("bitwise two-checkpoint asynchronous prefetch PASS", flush=True)


def check_arena_slabs():
    before = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    arena = _Arena(3072, max_slab_bytes=2048)
    after = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    assert after - before == 3072
    assert [buffer.numel() for buffer in arena.buffers] == [2048, 1024]
    a, slot_a = arena.allocate(torch.empty(900, dtype=torch.uint8))
    b, slot_b = arena.allocate(torch.empty(900, dtype=torch.uint8))
    c, slot_c = arena.allocate(torch.empty(900, dtype=torch.uint8))
    assert slot_a[0] == slot_b[0] == 0 and slot_c[0] == 1
    a.fill_(17); b.fill_(29); c.fill_(43)
    assert a.eq(17).all() and b.eq(29).all() and c.eq(43).all()
    assert arena.allocate(torch.empty(256, dtype=torch.uint8)) is None
    arena.release(slot_b); arena.release(slot_a); arena.release(slot_c)
    assert arena.free == [[(0, 2048)], [(0, 1024)]]
    assert arena.allocate(torch.empty(2500, dtype=torch.uint8)) is None
    print("exact pinned-byte budget, separate slabs, coalescing and fragmentation PASS", flush=True)
    # Two 131584-byte outputs occupy different 256-KiB slabs. Exercise real
    # asynchronous H2D prefetch/release with that layout and strict gradients.
    with patch("frozen_linear_cache._Arena", side_effect=lambda size: _Arena(size, max_slab_bytes=256 * 1024)):
        check_prefetch()


if __name__ == "__main__":
    torch.manual_seed(1234)
    check((33750, 5376, 7168), cpu=False)
    for shape in ((33750, 5376, 7168), (33750, 7168, 5376),
                  (33750, 5376, 28672), (33750, 14336, 5376), (3, 2688, 96768)):
        check(shape, cpu=True, bias=shape[0] == 3)
    check((3, 2688, 5376), cpu=True, dtype=torch.float32, bias=True)
    check((257, 128, 256), cpu=True, entry_gib=0)
    check_prefetch()
    check_arena_slabs()

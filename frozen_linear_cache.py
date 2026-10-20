"""Retain frozen GEMM outputs using native selective checkpoint replay.

The dispatch modes surround only Linear, so compiled RMSNorm never executes
under a mode. A fixed pinned arena bounds host allocations across input shapes.
"""

from collections import Counter
import os
from pathlib import Path
from types import MethodType
import weakref

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from attention_checkpoint import _active

_OPS = [torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default]


def _available_memory():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    candidates = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, name = line.split(":", 2)
        if "memory" in controllers.split(",") or not controllers:
            candidates += [base / name.lstrip("/") for base in candidates[:2]]
    for base in candidates:
        for limit_name, used_name in (("memory.max", "memory.current"),
                                      ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            try:
                limit = (base / limit_name).read_text().strip()
                used = int((base / used_name).read_text())
                if limit != "max" and int(limit) < 2**60:
                    available = min(available, max(0, int(limit) - used))
            except (OSError, ValueError):
                pass
    return available


class _Arena:
    def __init__(self, size):
        self.size = size // 256 * 256
        self.buffer = torch.empty(self.size, dtype=torch.uint8, pin_memory=True)
        self.free = [(0, self.size)]

    def allocate(self, value):
        size = value.numel() * value.element_size()
        reserved = (size + 255) // 256 * 256
        for index, (offset, length) in enumerate(self.free):
            if length >= reserved:
                self.free[index:index + 1] = [(offset + reserved, length - reserved)] if length > reserved else []
                view = self.buffer.narrow(0, offset, size).view(value.dtype).reshape(value.shape)
                return view, (offset, reserved)
        return None

    def release(self, slot):
        ranges = sorted([*self.free, slot])
        merged = []
        for offset, length in ranges:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                start, previous = merged.pop()
                merged.append((start, previous + length))
            else:
                merged.append((offset, length))
        self.free = merged


class _Record:
    def __init__(self, owner, value, host=None, slot=None):
        self.owner, self.host, self.slot = owner, host, slot
        self.size = value.numel() * value.element_size()
        self.gpu = value.detach() if host is None else None
        self.version = value._version
        self.ready = self.read_done = None
        self.replayed = False
        self.released = False
        if host is not None:
            owner.d2h.wait_stream(torch.cuda.current_stream(owner.device))
            with torch.cuda.stream(owner.d2h):
                host.copy_(value.detach(), non_blocking=True)
                value.record_stream(owner.d2h)
                self.ready = torch.cuda.Event()
                self.ready.record(owner.d2h)

    def materialize(self, wait=False):
        if self.gpu is None:
            with torch.cuda.stream(self.owner.h2d):
                self.owner.h2d.wait_event(self.ready)
                self.gpu = torch.empty_like(self.host, device=self.owner.device)
                self.gpu.copy_(self.host, non_blocking=True)
                self.read_done = torch.cuda.Event()
                self.read_done.record(self.owner.h2d)
                self.version = self.gpu._version
        if wait and self.read_done is not None:
            stream = torch.cuda.current_stream(self.owner.device)
            stream.wait_event(self.read_done)
            self.gpu.record_stream(stream)
        return self.gpu

    def get_val(self, allow_cache_entry_mutation=False):
        value = self.materialize(wait=True)
        if not allow_cache_entry_mutation and value._version != self.version:
            raise RuntimeError("frozen Linear cache output was mutated")
        self.replayed = True
        self.owner.counts["replayed_ops"] += 1
        return value

    def release(self):
        if self.released:
            return
        self.released = True
        if self.host is not None:
            # Any later D2H write into a reused arena slot waits for this read.
            if self.read_done is not None:
                self.owner.d2h.wait_event(self.read_done)
            self.owner.arena.release(self.slot)
            self.owner.cpu_live -= self.size
        else:
            self.owner.gpu_live -= self.size
        if not self.replayed:
            self.owner.counts["discarded_ops"] += 1
        self.host = self.gpu = None


class _Block:
    def __init__(self, owner):
        self.owner = owner
        self.calls = {}
        self.positions = Counter()
        self.records = []
        self.replaying = self.cleared = False
        owner.blocks.append(weakref.ref(self))

    def capture(self, module, input):
        capture, replay = create_selective_checkpoint_contexts(_OPS)
        with capture:
            output = F.linear(input, module.weight, module.bias)
        records = []
        valid = bool(capture.storage)
        for operation, values in capture.storage.items():
            for index, wrapped in enumerate(values):
                record = self.owner.pack(wrapped.get_val(False))
                if record is None:
                    valid = False
                    break
                values[index] = record
                records.append(record)
            if not valid:
                break
        if not valid:
            for record in records:
                record.release()
            replay.storage.clear()
            call = None
            self.owner.counts["budget_misses"] += 1
        else:
            call = replay
            self.records.extend(records)
            for record in records:
                self.owner.counts["saved_cpu_ops" if record.host is not None else "saved_gpu_ops"] += 1
        self.calls.setdefault(id(module), []).append(call)
        return output

    def replay(self, module, input):
        self.replaying = True
        position = self.positions[id(module)]
        self.positions[id(module)] += 1
        call = self.calls[id(module)][position]
        if call is None:
            self.owner.counts["recomputed_calls"] += 1
            return F.linear(input, module.weight, module.bias)
        with call:
            return F.linear(input, module.weight, module.bias)

    def clear(self):
        if self.cleared:
            return
        self.cleared = True
        for record in self.records:
            record.release()
        self.records.clear()
        self.calls.clear()
        if self.replaying:
            self.owner.prefetch()


class LinearCache:
    def __init__(self, device, gpu_gib, cpu_gib, local_world_size=None, max_entry_gib=1):
        if min(gpu_gib, cpu_gib, max_entry_gib) < 0:
            raise ValueError("Linear cache budgets must be nonnegative")
        self.device = device
        ranks = local_world_size or int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self.gpu_limit = int(gpu_gib * 1024**3)
        self.cpu_limit = min(int(cpu_gib * 1024**3), int(_available_memory() * 0.85 / ranks))
        self.max_entry_bytes = int(max_entry_gib * 1024**3)
        self.gpu_live = self.cpu_live = self.gpu_peak = self.cpu_peak = 0
        self.arena = self.d2h = self.h2d = None
        self.blocks = []
        self.counts = Counter()
        self.modules = 0

    def pack(self, value):
        if not value.is_contiguous():
            return None
        size = value.numel() * value.element_size()
        if size > self.max_entry_bytes:
            self.counts["oversized_ops"] += 1
            return None
        if self.gpu_live + size <= self.gpu_limit:
            self.gpu_live += size
            self.gpu_peak = max(self.gpu_peak, self.gpu_live)
            return _Record(self, value)
        if self.cpu_live + size > self.cpu_limit or self.cpu_limit == 0:
            return None
        if self.arena is None:
            self.arena = _Arena(self.cpu_limit)
            self.d2h = torch.cuda.Stream(device=self.device)
            self.h2d = torch.cuda.Stream(device=self.device)
        allocation = self.arena.allocate(value)
        if allocation is None:
            return None
        host, slot = allocation
        self.cpu_live += size
        self.cpu_peak = max(self.cpu_peak, self.cpu_live)
        return _Record(self, value, host, slot)

    def prefetch(self):
        self.blocks = [reference for reference in self.blocks if reference() is not None]
        for reference in reversed(self.blocks):
            block = reference()
            if not block.cleared:
                for record in block.records:
                    if record.host is not None:
                        record.materialize()
                self.counts["prefetched_blocks"] += 1
                break

    def report(self):
        return {"modules": self.modules, **self.counts,
                "gpu_limit": self.gpu_limit, "cpu_limit": self.cpu_limit,
                "max_entry_bytes": self.max_entry_bytes,
                "gpu_peak": self.gpu_peak, "cpu_peak": self.cpu_peak,
                "gpu_live": self.gpu_live, "cpu_live": self.cpu_live,
                "pinned_arena_bytes": self.arena.size if self.arena else 0,
                "cuda_max_allocated": torch.cuda.max_memory_allocated(self.device),
                "cuda_max_reserved": torch.cuda.max_memory_reserved(self.device)}


def _forward(module, input):
    active = _active.get()
    if active is None or module.weight.requires_grad or (module.bias is not None and module.bias.requires_grad):
        return F.linear(input, module.weight, module.bias)
    cache, replay = active
    owner = module._linear_cache
    key = ("frozen_linear", id(owner))
    if key not in cache.extensions:
        cache.extensions[key] = _Block(owner)
    block = cache.extensions[key]
    return block.replay(module, input) if replay else block.capture(module, input)


def install(model, gpu_gib=16, cpu_gib=64, max_entry_gib=1):
    owner = LinearCache(next(model.parameters()).device, gpu_gib, cpu_gib,
                        max_entry_gib=max_entry_gib)
    for module in model.modules():
        if type(module) is torch.nn.Linear and not module.weight.requires_grad and (
                module.bias is None or not module.bias.requires_grad):
            module._linear_cache = owner
            module.forward = MethodType(_forward, module)
            owner.modules += 1
    return owner

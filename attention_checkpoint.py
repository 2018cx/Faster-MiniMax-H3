"""Retain cuDNN attention results across a DeepSpeed activation checkpoint.

Only attention outputs and softmax statistics survive the first forward. Q/K/V
are recomputed normally and the original cuDNN backward consumes them. The
cache belongs to one checkpoint invocation, never to a model or training step.
"""

from contextvars import ContextVar

import torch
from torch.nn import functional as F
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    _apply_rotary_emb,
)

_active = ContextVar("attention_checkpoint", default=None)


class _SavedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, record):
        out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _ = record
        ctx.save_for_backward(query, key, value, out, lse, cum_q, cum_k, seed, offset)
        ctx.max_q, ctx.max_k = max_q, max_k
        return out

    @staticmethod
    def backward(ctx, grad):
        query, key, value, out, lse, cum_q, cum_k, seed, offset = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            grad.contiguous(), query, key, value, out, lse, seed, offset,
            None, cum_q, cum_k, ctx.max_q, ctx.max_k, 0.0, False,
        )
        return dq, dk, dv, None


class _Cache:
    def __init__(self, owner):
        self.owner = owner
        self.records = {}
        self.bytes = 0
        self.extensions = {}

    def clear(self):
        self.owner.retained -= self.bytes
        self.bytes = 0
        self.records.clear()
        extensions, self.extensions = self.extensions, {}
        for extension in extensions.values():
            extension.clear()

    def __del__(self):
        self.clear()


class AttentionCheckpoint:
    def __init__(self, checkpoint, max_bytes):
        self.checkpoint = checkpoint
        self.max_bytes = max_bytes
        self.retained = 0
        self.saved = self.replayed = self.recomputed = 0
        self.peak_bytes = 0

    def __call__(self, function, *args):
        cache = _Cache(self)

        def run(*inputs):
            replay = torch.is_grad_enabled()
            token = _active.set((cache, replay))
            try:
                return function(*inputs)
            finally:
                _active.reset(token)
                if replay:
                    cache.clear()

        return self.checkpoint(run, *args)


class CachedCudnnProcessor(MiniMaxH3AttnProcessor):
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        active = _active.get()
        if (active is None or attention_mask is not None or self._parallel_config is not None
                or self._attention_backend != "_native_cudnn"):
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        cache, replay = active
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))

        if replay and id(self) in cache.records:
            hidden_states = _SavedAttention.apply(query, key, value, cache.records.pop(id(self)))
            cache.owner.replayed += 1
        elif (not replay and cache.owner.retained + query.numel() * query.element_size()
              + query.numel() // query.shape[-1] * 4 <= cache.owner.max_bytes):
            record = torch.ops.aten._scaled_dot_product_cudnn_attention(
                query, key, value, None, True, 0.0, False,
            )
            hidden_states = record[0]
            cache.records[id(self)] = record
            size = sum(x.numel() * x.element_size() for x in record if isinstance(x, torch.Tensor))
            cache.bytes += size
            cache.owner.retained += size
            cache.owner.peak_bytes = max(cache.owner.peak_bytes, cache.owner.retained)
            cache.owner.saved += 1
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            cache.owner.recomputed += 1

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


def install(model, checkpoint, max_gib=32):
    processors = {}
    for name, original in model.attn_processors.items():
        processor = CachedCudnnProcessor()
        processor._attention_backend = original._attention_backend
        processor._parallel_config = original._parallel_config
        processors[name] = processor
    model.set_attn_processor(processors)
    return AttentionCheckpoint(checkpoint, int(max_gib * 1024**3))

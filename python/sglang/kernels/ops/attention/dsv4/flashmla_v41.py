"""torch.ops.sgl_flashmla_v41: deepseek-ai/FlashMLA@2e5429f's SM100 V4.1 kernels, which read the
token-interleaved KV cache (see kv_layout.py). Built by sglang/kernels/aot/flashmla_v41/build.sh.

The library is looked up in order: ``$SGLANG_FLASHMLA_V41_LIB``, ``_flashmla_v41/`` next to this
file, then the installed sgl_kernel package.
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Optional, Tuple

import torch

_LIB_NAME = "libflashmla_v41_ops.so"


def _candidates():
    env = os.environ.get("SGLANG_FLASHMLA_V41_LIB")
    if env:
        yield env
    yield os.path.join(os.path.dirname(__file__), "_flashmla_v41", _LIB_NAME)
    try:
        import sgl_kernel

        root = os.path.dirname(sgl_kernel.__file__)
        yield os.path.join(root, _LIB_NAME)
        yield from glob.glob(os.path.join(root, "flashmla_v41_ops*.so"))
    except ImportError:
        pass


@functools.cache
def load() -> None:
    if hasattr(torch.ops.sgl_flashmla_v41, "mega_decode"):
        return
    for path in _candidates():
        if os.path.exists(path):
            torch.ops.load_library(path)
            return
    raise ImportError(
        "The token-interleaved DeepSeek-V4.1 KV cache needs flashmla_v41_ops "
        "(sglang/kernels/aot/flashmla_v41/build.sh); set SGLANG_FLASHMLA_V41_LIB to its path"
    )


def is_available() -> bool:
    try:
        load()
        return True
    except (ImportError, OSError):
        return False


def mega_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    load()
    return tuple(
        torch.ops.sgl_flashmla_v41.mega_decode.default(
            q,
            kv,
            indices,
            positions,
            cos_sin_cache,
            softmax_scale,
            q.shape[1] // 8,
            attn_sink,
            topk_length,
            extra_kv,
            extra_indices,
            extra_topk_length,
        )
    )


def mega_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    load()
    return tuple(
        torch.ops.sgl_flashmla_v41.mega_prefill.default(
            q,
            kv,
            indices,
            positions,
            cos_sin_cache,
            softmax_scale,
            q.shape[1] // 8,
            attn_sink,
            topk_length,
        )
    )


def sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    head_dim_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [b, s_q, h_q, 512] bf16, kv [pages, page_size, 1, 528], indices [b, s_q, topk] int32;
    returns (out [b, s_q, h_q, 512], lse [b, h_q, s_q])."""
    load()
    return torch.ops.sgl_flashmla_v41.sparse_decode_fwd.default(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        extra_kv,
        extra_indices,
        extra_topk_length,
        head_dim_v,
        softmax_scale,
    )

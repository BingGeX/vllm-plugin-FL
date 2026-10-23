# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Triton-free KV block zeroing for the Kunlunxin XPU backend.

vLLM 0.28 zeroes freshly allocated KV cache blocks inside ``KVBlockZeroer``
via the ``_zero_kv_blocks_kernel`` Triton kernel, which dereferences raw
absolute addresses held in a uint64 tensor::

    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))

``patches.triton_compat`` only made that kernel *compile* on the XPU backend.
Launching it still kills the device on P800 at the first real request::

    xpuLaunchKernel("_zero_kv_blocks_kernel", ...)
      -> Exception in kernel execution (err_code: -714)

so ``POST /v1/chat/completions`` answered HTTP 500.  Reimplement
``zero_block_ids`` with ordinary tensor ops instead.

Semantics are preserved, not weakened.  Upstream builds one segment per
(outer index, virtual index): segment start = ``data_ptr + outer_offset +
virtual_index * block_stride_bytes``, block stride = ``block_stride_bytes *
ratio``, zeroed span = one kernel page (one kernel block's K/V bytes).  Hence
physical kernel block ``block_id * ratio + virtual_index``.  ``index_fill_``
over exactly that index set zeroes the same bytes for every block vLLM hands
us, so a freshly allocated block still reads back as zeros.

TODO: remove once the XPU Triton backend supports absolute-address pointers
or vLLM ships a non-Triton zeroing path.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def _collect_kv_segments(
    attn_groups,
    kernel_block_sizes,
    cache_dtype,
    runner_only_attn_layers,
    static_forward_context,
):
    """Map each distinct KV cache tensor to ``(tensor, block_dim, ratio)``.

    Mirrors ``KVBlockZeroer.__init__`` so exactly the same tensors, block
    dimension and virtual-block ratio are covered.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    segments = []
    seen_ptrs = set()
    for group in attn_groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, AttentionSpec):
            # Mamba/SSM state groups are skipped upstream too.
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
        if not kernel_bs or spec.block_size % kernel_bs:
            continue
        ratio = spec.block_size // kernel_bs
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_bs,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        for layer_name in group.layer_names:
            if layer_name in runner_only_attn_layers:
                continue
            kv = static_forward_context[layer_name].kv_cache
            if not isinstance(kv, torch.Tensor):
                continue
            ptr = kv.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            if kv.shape[block_dim] % ratio:
                continue
            segments.append((kv, block_dim, ratio))
    return segments


def _zero_block_ids_pytorch(self, block_ids) -> None:
    """Triton-free replacement for ``KVBlockZeroer.zero_block_ids``."""
    segments = getattr(self, "_kunlunxin_kv_segments", None)
    if not block_ids or not segments:
        return
    ids = [int(b) for b in block_ids]
    for kv, block_dim, ratio in segments:
        num_kernel_blocks = kv.shape[block_dim]
        physical = []
        for block_id in ids:
            start = block_id * ratio
            if start >= 0 and start + ratio <= num_kernel_blocks:
                physical.extend(range(start, start + ratio))
        if not physical:
            continue
        index = torch.tensor(physical, dtype=torch.long, device=kv.device)
        kv.index_fill_(block_dim, index, 0)


def apply_kv_block_zero_compat() -> None:
    """Replace the Triton KV-block zeroing kernel with tensor-op zeroing."""
    global _patched
    if _patched:
        return
    try:
        from vllm.v1.worker.utils import KVBlockZeroer
    except Exception:  # pragma: no cover - vLLM without the zeroer
        return

    if getattr(KVBlockZeroer.zero_block_ids, "_kunlunxin_pytorch_zero", False):
        _patched = True
        return

    original_init = KVBlockZeroer.__init__

    def init_and_collect(
        self,
        device,
        attn_groups_iter,
        kernel_block_sizes,
        cache_dtype,
        static_forward_context,
        runner_only_attn_layers=None,
    ):
        # Materialize first: upstream consumes the iterable.
        groups = list(attn_groups_iter)
        original_init(
            self,
            device,
            iter(groups),
            kernel_block_sizes,
            cache_dtype,
            static_forward_context,
            runner_only_attn_layers,
        )
        try:
            self._kunlunxin_kv_segments = _collect_kv_segments(
                groups,
                kernel_block_sizes,
                cache_dtype,
                runner_only_attn_layers or set(),
                static_forward_context,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "kunlunxin kv_block_zero_compat: segment collection failed: %s", exc
            )
            self._kunlunxin_kv_segments = None

    _zero_block_ids_pytorch._kunlunxin_pytorch_zero = True
    KVBlockZeroer.__init__ = init_and_collect
    KVBlockZeroer.zero_block_ids = _zero_block_ids_pytorch
    _patched = True
    logger.info(
        "kunlunxin kv_block_zero_compat: KV block zeroing uses tensor ops "
        "instead of the Triton absolute-address kernel"
    )

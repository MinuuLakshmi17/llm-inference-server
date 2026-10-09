"""Symmetric INT8 weight-only quantization with real integer GEMM.

Weights are quantized once, statically::

    scale_w = max_abs(W) / 127            # per output channel (or per tensor)
    W_q     = round(clamp(W / scale_w, -128, 127)).to(int8)

Activations are quantized dynamically, per token, on every forward pass::

    scale_x = max_abs(x_token) / 127
    x_q     = round(clamp(x / scale_x, -128, 127)).to(int8)

The matmul runs in integer arithmetic and is descaled once::

    y = (x_q @ W_q).to(float) * (scale_x * scale_w) + b      # int32 accumulate

``torch._int_mm`` accumulates int8*int8 products in int32, which is exact
for the hidden sizes used here (worst case 12288 * 128 * 128 < 2**31).

Scope and honesty bounds, stated plainly:
  - Only linear projection weights are quantized (nn.Linear and HF Conv1D).
    Token embeddings, biases, layer norms and the KV cache stay in float.
  - Layers whose weights are *tied* to a non-quantized module (GPT-2's
    lm_head shares storage with the token embedding) are left in float
    rather than double-counting or silently forking the shared tensor.
  - This is not a fused kernel: per-token activation quantization adds
    overhead, so the speedup is real but smaller than a production
    quantized kernel would achieve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:  # HF GPT-2 uses Conv1D instead of nn.Linear for its projections.
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - defensive
    Conv1D = ()

_LINEAR_TYPES = (nn.Linear,) + ((Conv1D,) if Conv1D else ())


@dataclass
class QuantStats:
    num_linear_layers: int = 0
    num_skipped_tied: int = 0
    original_bytes: int = 0
    quantized_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes == 0:
            return 0.0
        return self.original_bytes / self.quantized_bytes


def quantize_symmetric(
    weight: torch.Tensor, per_channel: bool = True, channel_dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float weight matrix to int8 with symmetric scaling.

    Returns ``(q, scale)`` where ``q`` is int8. ``scale`` is a scalar for
    per-tensor mode, or has one entry per output channel otherwise. The
    caller decides which dim holds output channels via ``channel_dim``.
    """
    w = weight.detach().float()
    if per_channel:
        dims = [d for d in range(w.dim()) if d != channel_dim]
        max_abs = w.abs().amax(dim=dims, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
        q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    else:
        max_abs = w.abs().max()
        scale = max_abs / 127.0 if max_abs > 0 else torch.tensor(1.0)
        q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return q, scale


def quantize_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric INT8 quantization, one scale per row (token)."""
    x = x.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q, scale


def probe_outlier_channels(
    model: nn.Module, module_name: str, probe_ids: torch.Tensor, threshold: float = 6.0
) -> torch.Tensor:
    """Find input channels whose magnitude exceeds ``threshold``.

    Runs ``probe_ids`` through the (still fp32) model, hooks the input of
    ``module_name``, and returns the sorted indices of channels whose
    abs-max over the probe exceeds the threshold. Activation outliers in
    transformers are a stable model property (same channels across tokens),
    so a short probe is sufficient. Follows the LLM.int8() decomposition.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, inputs, _out):
        captured["x"] = inputs[0].detach()

    handle = dict(model.named_modules())[module_name].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(probe_ids)
    finally:
        handle.remove()
    x = captured["x"].float().reshape(-1, captured["x"].shape[-1])
    channel_max = x.abs().amax(dim=0)
    return torch.where(channel_max > threshold)[0].sort().values


class Int8Linear(nn.Module):
    """Drop-in INT8 replacement for ``nn.Linear`` and HF ``Conv1D``.

    The weight is stored once as int8 in ``[in_features, out_features]``
    layout (transposed at construction for ``nn.Linear``) plus a per-output-
    channel scale row. ``weight_q``/``scale`` are buffers so ``.to(device)``
    and ``state_dict`` keep working; the bias stays a float parameter.

    Activations are quantized dynamically per token each forward pass, and
    the matmul runs as integer GEMM into int32. When ``outlier_idx`` is
    given, those input channels bypass quantization entirely: they are
    multiplied in float against a fp32 copy of their weight columns
    (the LLM.int8() decomposition). This handles layers whose inputs carry
    persistent outlier channels -- e.g. the output head sitting on the
    final hidden state, where 3 of 768 channels reach magnitudes of 40-150
    while the median channel sits at 0.6. A per-token scale would crush
    every other channel into a handful of int8 levels.
    """

    def __init__(self, module: nn.Module, per_channel: bool = True,
                 outlier_idx: torch.Tensor | None = None):
        super().__init__()
        if isinstance(module, nn.Linear):
            # nn.Linear weight is [out, in]; we want [in, out].
            w_eff = module.weight.data.t().contiguous()
        else:  # Conv1D: weight is already [in, out], forward is x @ W + b.
            w_eff = module.weight.data.contiguous()
        in_features = w_eff.shape[0]
        if outlier_idx is not None and len(outlier_idx):
            keep = torch.ones(in_features, dtype=torch.bool)
            keep[outlier_idx] = False
            self.register_buffer("keep_idx", torch.where(keep)[0])
            self.register_buffer("outlier_idx", outlier_idx.sort().values)
            # fp32 copy of just the outlier columns (a handful, not the matrix).
            self.register_buffer("w_outlier", w_eff[outlier_idx].float().contiguous())
            w_eff = w_eff[keep]
        else:
            self.register_buffer("keep_idx", torch.arange(in_features))
            self.register_buffer("outlier_idx", torch.empty(0, dtype=torch.long))
            self.register_buffer("w_outlier", torch.empty(0, 0))
        q, scale = quantize_symmetric(w_eff, per_channel=per_channel, channel_dim=1)
        self.register_buffer("weight_q", q)  # [in_keep, out] int8
        # Broadcastable row vector [1, out] for the descaling multiply.
        self.register_buffer("scale", scale.reshape(1, -1))
        if module.bias is not None:
            self.bias = nn.Parameter(module.bias.data.clone())
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features = w_eff.shape[1]
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        x_q, x_scale = quantize_activation_per_token(x2d[:, self.keep_idx])
        # int8 x int8 -> int32, then a single float descale.
        acc = torch._int_mm(x_q, self.weight_q)
        y = acc.float() * (x_scale * self.scale)
        if len(self.outlier_idx):
            y = y + x2d[:, self.outlier_idx].float() @ self.w_outlier
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(orig_shape[:-1] + (self.out_features,))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        mode = "per-channel" if self.per_channel else "per-tensor"
        out = f", outliers={len(self.outlier_idx)}" if len(self.outlier_idx) else ""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"int8-{mode}, int-gemm{out}")

    @classmethod
    def _from_parts(cls, weight_q: torch.Tensor, scale: torch.Tensor,
                    bias: torch.Tensor | None, per_channel: bool,
                    outlier_idx: torch.Tensor | None = None,
                    w_outlier: torch.Tensor | None = None) -> "Int8Linear":
        """Build from a pre-quantized [in, out] int8 weight and [1, out] scale.

        ``outlier_idx``/``w_outlier`` optionally describe the fp32 side path
        (see the class docstring); ``w_outlier`` is [n_outlier, out] fp32.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        in_features = weight_q.shape[0] + (len(outlier_idx) if outlier_idx is not None and len(outlier_idx) else 0)
        if outlier_idx is not None and len(outlier_idx):
            keep = torch.ones(in_features, dtype=torch.bool)
            keep[outlier_idx] = False
            obj.register_buffer("keep_idx", torch.where(keep)[0])
            obj.register_buffer("outlier_idx", outlier_idx.sort().values)
            obj.register_buffer("w_outlier", w_outlier.contiguous())
        else:
            obj.register_buffer("keep_idx", torch.arange(weight_q.shape[0]))
            obj.register_buffer("outlier_idx", torch.empty(0, dtype=torch.long))
            obj.register_buffer("w_outlier", torch.empty(0, 0))
        obj.register_buffer("weight_q", weight_q.contiguous())
        obj.register_buffer("scale", scale.reshape(1, -1))
        obj.bias = nn.Parameter(bias.clone()) if bias is not None else None
        obj.in_features = in_features
        obj.out_features = weight_q.shape[1]
        obj.per_channel = per_channel
        return obj


class Int8Embedding(nn.Module):
    """Drop-in INT8 replacement for ``nn.Embedding``.

    Each vocabulary row gets its own scale (per-row === per-channel with
    ``channel_dim=0``). Lookup gathers int8 rows and descsales them; the
    memory traffic per lookup drops 4x, which is the whole point for
    memory-bound generation.
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        q, scale = quantize_symmetric(
            embedding.weight.data, per_channel=True, channel_dim=0
        )
        self.register_buffer("weight_q", q)      # [vocab, dim] int8
        self.register_buffer("scale", scale)     # [vocab, 1]
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self.weight_q[input_ids]
        return rows.float() * self.scale[input_ids]

    @classmethod
    def _from_parts(cls, weight_q: torch.Tensor, scale: torch.Tensor) -> "Int8Embedding":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.register_buffer("weight_q", weight_q)
        obj.register_buffer("scale", scale)
        obj.num_embeddings, obj.embedding_dim = weight_q.shape
        return obj


def _parent_module(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def quantize_model_(
    model: nn.Module,
    per_channel: bool = True,
    skip_names: tuple[str, ...] = (),
    outlier_probe_ids: torch.Tensor | None = None,
    outlier_threshold: float = 6.0,
) -> QuantStats:
    """Replace linear projections (and tied embedding/head pairs) with INT8.

    Handles ``nn.Linear`` and HF ``Conv1D``. A linear layer whose weight is
    *tied* to an ``nn.Embedding`` (GPT-2's lm_head <-> token embedding) is
    not skipped: the shared ``[vocab, dim]`` tensor is quantized once,
    per-row, and both consumers are rewired to it -- the embedding becomes
    :class:`Int8Embedding`, the head an :class:`Int8Linear` over the
    transposed int8 data. If ``outlier_probe_ids`` is given, the head's
    input is probed for outlier channels (LLM.int8() decomposition): those
    columns stay in fp32 alongside the int8 GEMM. Weights tied to anything
    else stay in float rather than being forked and double-counted. Returns
    byte counts over the replaced weights so callers can report the real
    saving.
    """
    stats = QuantStats()
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, _LINEAR_TYPES)
        and not any(skip in name for skip in skip_names)
    ]

    def replace(name: str, new_module: nn.Module) -> None:
        parent, attr = _parent_module(model, name)
        setattr(parent, attr, new_module)

    # Identify the tied embedding/head pair up front so the outlier probe
    # runs on the untouched fp32 model.
    tied_pair: tuple[str, str, nn.Module] | None = None
    for name, module in targets:
        w_ptr, w_numel = module.weight.data_ptr(), module.weight.numel()
        for ename, emod in model.named_modules():
            if isinstance(emod, nn.Embedding) and emod.weight.data_ptr() == w_ptr \
                    and emod.weight.numel() == w_numel:
                tied_pair = (ename, name, module)
                break
        if tied_pair is not None:
            break
    head_outliers: torch.Tensor | None = None
    if tied_pair is not None and outlier_probe_ids is not None:
        head_outliers = probe_outlier_channels(
            model, tied_pair[1], outlier_probe_ids, outlier_threshold
        )

    handled: set[int] = set()  # weight ids already quantized (tied pairs)
    for name, module in targets:
        wid = module.weight.data_ptr()
        if wid in handled:
            continue
        if tied_pair is not None and name == tied_pair[1]:
            emb_name, _, _ = tied_pair
            # Quantize the shared [vocab, dim] tensor once, per row.
            w_fp32 = module.weight.data
            q, scale = quantize_symmetric(
                w_fp32, per_channel=True, channel_dim=0
            )
            replace(emb_name, Int8Embedding._from_parts(q, scale))
            # Head: transposed int8 for the GEMM + fp32 outlier side path.
            w_eff_q = q.t().contiguous()  # [dim, vocab]
            outlier_idx, w_outlier = None, None
            if head_outliers is not None and len(head_outliers):
                outlier_idx = head_outliers
                keep = torch.ones(w_eff_q.shape[0], dtype=torch.bool)
                keep[outlier_idx] = False
                w_outlier = w_fp32.t().contiguous()[outlier_idx].float()
                w_eff_q = w_eff_q[keep]
            replace(
                name,
                Int8Linear._from_parts(
                    w_eff_q, scale.reshape(1, -1),
                    module.bias.data if module.bias is not None else None,
                    per_channel, outlier_idx=outlier_idx, w_outlier=w_outlier,
                ),
            )
            handled.add(wid)
            stats.num_linear_layers += 1
            stats.original_bytes += module.weight.numel() * module.weight.element_size()
            stats.quantized_bytes += (
                q.numel() * q.element_size()          # embedding copy
                + w_eff_q.numel() * w_eff_q.element_size()  # transposed GEMM copy
                + scale.numel() * scale.element_size()
                + (w_outlier.numel() * w_outlier.element_size() if w_outlier is not None else 0)
            )
            continue
        # Tied to a non-embedding module: leave in float, don't fork.
        is_tied = False
        for m in model.modules():
            if isinstance(m, (_LINEAR_TYPES, nn.Embedding)) or m is module:
                continue
            for p in m.parameters(recurse=False):
                if p.data_ptr() == w_ptr and p.numel() == w_numel:
                    is_tied = True
                    break
            if is_tied:
                break
        if is_tied:
            stats.num_skipped_tied += 1
            continue
        parent, attr = _parent_module(model, name)
        quantized = Int8Linear(module, per_channel=per_channel)
        setattr(parent, attr, quantized)
        stats.num_linear_layers += 1
        stats.original_bytes += module.weight.numel() * module.weight.element_size()
        stats.quantized_bytes += (
            quantized.weight_q.numel() * quantized.weight_q.element_size()
            + quantized.scale.numel() * quantized.scale.element_size()
        )
    return stats


def model_weight_bytes(model: nn.Module) -> int:
    """Total bytes held by all parameters and buffers (int8-aware)."""
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total

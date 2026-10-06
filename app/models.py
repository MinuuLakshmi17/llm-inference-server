from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings
from .metrics import KV_BLOCK_FRAGMENTATION, KV_BLOCKS_TOTAL, KV_BLOCKS_USED
from .paged import BlockAllocator, PagedSequence, gather_sequence, store_prefill

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    token_ids: list[int]
    past_key_values: Any
    logits: torch.Tensor


class ModelRunner:
    """Model execution layer with a true batched decode path."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.dtype = self._resolve_dtype(settings.model_dtype)
        logger.info("Loading model=%s device=%s dtype=%s", settings.model_name, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {}
        if self.dtype is not None and self.device.type == "cuda":
            kwargs["torch_dtype"] = self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(settings.model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()

        # Paged KV cache (optional): sequences keep a block table instead of
        # one contiguous cache. See app/paged.py.
        self.allocator: BlockAllocator | None = None
        if settings.paged_kv:
            config = self.model.config
            num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", None))
            num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", None))
            head_dim = getattr(config, "n_embd", getattr(config, "hidden_size", None)) // num_heads
            param_dtype = next(self.model.parameters()).dtype
            self.allocator = BlockAllocator(
                num_blocks=settings.kv_num_blocks,
                block_size=settings.kv_block_size,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=param_dtype,
                device=self.device,
            )
            logger.info(
                "Paged KV cache enabled: %d blocks x %d tokens",
                settings.kv_num_blocks, settings.kv_block_size,
            )
            self._update_block_metrics()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested, but CUDA is unavailable.")
        return torch.device(requested)

    @staticmethod
    def _resolve_dtype(name: str):
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(name) or (_ for _ in ()).throw(ValueError(f"Unsupported MODEL_DTYPE={name}"))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @torch.inference_mode()
    def prefill(self, input_ids: list[int]) -> GenerationOutput:
        if self.allocator is not None:
            return self.prefill_paged(input_ids)
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        result = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        return GenerationOutput(input_ids, result.past_key_values, result.logits[:, -1, :])

    @torch.inference_mode()
    def prefill_paged(self, input_ids: list[int]) -> GenerationOutput:
        """Prefill, then chunk the contiguous KV cache into blocks."""
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        result = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        seq = store_prefill(self.allocator, result.past_key_values, len(input_ids))
        self._update_block_metrics()
        return GenerationOutput(input_ids, seq, result.logits[:, -1, :])

    @staticmethod
    def _pad_legacy_cache(caches: list[Any], lengths: list[int], device: torch.device):
        max_len = max(lengths)
        batched = []
        for layer in range(len(caches[0])):
            keys, values = [], []
            for cache, length in zip(caches, lengths):
                key, value = cache[layer]
                pad = max_len - length
                if pad:
                    key = torch.nn.functional.pad(key, (0, 0, pad, 0))
                    value = torch.nn.functional.pad(value, (0, 0, pad, 0))
                keys.append(key)
                values.append(value)
            batched.append((torch.cat(keys, dim=0).to(device), torch.cat(values, dim=0).to(device)))
        return tuple(batched)

    @torch.inference_mode()
    def decode_batch(self, token_ids: list[int], past_key_values: list[Any], cache_lengths: list[int]):
        """Run one transformer forward pass for every active sequence.

        In legacy mode ``past_key_values`` holds per-sequence contiguous
        caches; in paged mode it holds :class:`PagedSequence` objects and
        ``cache_lengths`` is ignored (each sequence knows its own length).
        """
        if self.allocator is not None:
            return self.decode_batch_paged(token_ids, past_key_values)
        if not token_ids or not (len(token_ids) == len(past_key_values) == len(cache_lengths)):
            raise ValueError("decode_batch requires equally-sized non-empty inputs")
        batch = len(token_ids)
        max_cache_len = max(cache_lengths)
        input_ids = torch.tensor([[t] for t in token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch, max_cache_len + 1), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((batch, 1), dtype=torch.long, device=self.device)
        for i, length in enumerate(cache_lengths):
            attention_mask[i, max_cache_len - length:] = 1
            position_ids[i, 0] = length
        batched_cache = self._pad_legacy_cache(past_key_values, cache_lengths, self.device)
        result = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=batched_cache, use_cache=True)
        per_sequence = []
        for i, old_length in enumerate(cache_lengths):
            new_length = old_length + 1
            start = max_cache_len + 1 - new_length
            layers = []
            for key, value in result.past_key_values:
                layers.append((key[i:i+1, :, start:, :], value[i:i+1, :, start:, :]))
            per_sequence.append(tuple(layers))
        return result.logits[:, -1, :], per_sequence

    @torch.inference_mode()
    def decode_batch_paged(
        self, token_ids: list[int], seqs: list["PagedSequence"]
    ) -> tuple[torch.Tensor, list["PagedSequence"]]:
        """One batched forward pass over paged sequences.

        Each sequence's blocks are gathered into a contiguous cache for the
        HF forward pass; afterwards the new token's K/V is scattered back
        into the sequence's current block slot.
        """
        if not token_ids or len(token_ids) != len(seqs):
            raise ValueError("decode_batch_paged requires equally-sized non-empty inputs")
        gathered = [gather_sequence(seq) for seq in seqs]
        cache_lengths = [length for _, length in gathered]
        batch = len(token_ids)
        max_cache_len = max(cache_lengths)
        input_ids = torch.tensor([[t] for t in token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch, max_cache_len + 1), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((batch, 1), dtype=torch.long, device=self.device)
        batched = []
        for layer in range(self.allocator.num_layers):
            keys, values = [], []
            for (layers, length) in gathered:
                key, value = layers[layer]
                pad = max_cache_len - length
                if pad:
                    key = torch.nn.functional.pad(key, (0, 0, pad, 0))
                    value = torch.nn.functional.pad(value, (0, 0, pad, 0))
                keys.append(key)
                values.append(value)
            batched.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        for i, length in enumerate(cache_lengths):
            attention_mask[i, max_cache_len - length :] = 1
            position_ids[i, 0] = length
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=tuple(batched),
            use_cache=True,
        )
        # The new token is the last valid position of each row.
        for i, seq in enumerate(seqs):
            per_layer = []
            for layer, (key, value) in enumerate(result.past_key_values):
                knew = key[i : i + 1, :, max_cache_len : max_cache_len + 1, :]
                vnew = value[i : i + 1, :, max_cache_len : max_cache_len + 1, :]
                per_layer.append((knew, vnew))
            seq.append_kv(per_layer)
        self._update_block_metrics()
        return result.logits[:, -1, :], seqs

    def release_state(self, state: Any) -> None:
        """Reclaim a finished request's blocks (no-op in legacy mode)."""
        if isinstance(state, PagedSequence):
            state.free()
            self._update_block_metrics()

    def _update_block_metrics(self) -> None:
        if self.allocator is None:
            return
        KV_BLOCKS_TOTAL.set(self.allocator.num_blocks)
        KV_BLOCKS_USED.set(self.allocator.num_used)
        KV_BLOCK_FRAGMENTATION.set(self.allocator.fragmentation())

    def sample(self, logits: torch.Tensor, temperature: float, top_k: int) -> int:
        logits = logits.float()
        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())
        logits = logits / temperature
        if top_k > 0:
            k = min(top_k, logits.shape[-1])
            values, _ = torch.topk(logits, k)
            logits = torch.where(logits < values[..., -1, None], torch.full_like(logits, float("-inf")), logits)
        return int(torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item())

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings

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
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        result = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        return GenerationOutput(input_ids, result.past_key_values, result.logits[:, -1, :])

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
        """Run one transformer forward pass for every active sequence."""
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

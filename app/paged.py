"""Paged KV-cache block management (vLLM-style).

The physical KV cache is divided into fixed-size *blocks*. Each sequence owns
a *block table* mapping logical block indices to physical block ids, plus a
logical token count. The allocator owns every physical block and reclaims
them when a sequence finishes.

This is the *management* layer: allocation, block tables, reclamation, and
fragmentation accounting. The transformer itself still runs on contiguous
tensors: before each forward pass a sequence's blocks are gathered into one
contiguous cache, and afterwards the new token's K/V is scattered back into
its block slot. A custom block-sparse attention kernel would remove the
gather step; that is the natural next extension.
"""

from __future__ import annotations

import torch


class BlockAllocator:
    """Pool of fixed-size physical KV blocks.

    Block ``i`` holds ``block_size`` token slots of (key, value) for every
    transformer layer. Blocks are materialized lazily on first write so an
    idle pool costs no tensor memory.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self._free: list[int] = list(range(num_blocks))
        # block_id -> per-layer [(key, value)] or None (allocated, not yet written)
        self._blocks: dict[int, list[tuple[torch.Tensor, torch.Tensor]] | None] = {}
        # block_id -> number of token slots written so far
        self._slots: dict[int, int] = {}
        self._tokens_stored = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def fragmentation(self) -> float:
        """Share of allocated block capacity holding no token: 0.0 = perfect packing."""
        capacity = self.num_used * self.block_size
        if capacity == 0:
            return 0.0
        return 1.0 - self._tokens_stored / capacity

    def allocate(self) -> int:
        """Take one physical block from the free pool (LIFO for cache warmth)."""
        if not self._free:
            raise RuntimeError(
                f"KV block pool exhausted ({self.num_blocks} blocks). "
                "Increase KV_NUM_BLOCKS or lower MAX_BATCH_SIZE / MAX_NEW_TOKENS."
            )
        block_id = self._free.pop()
        self._blocks[block_id] = None
        self._slots[block_id] = 0
        return block_id

    def free(self, block_id: int) -> None:
        entry = self._blocks.pop(block_id, None)
        if entry is None and block_id not in self._slots:
            return  # already free / unknown; ignore defensively
        self._tokens_stored -= self._slots.pop(block_id, 0)
        self._free.append(block_id)

    def _materialize(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.zeros(
                    1, self.num_heads, self.block_size, self.head_dim,
                    dtype=self.dtype, device=self.device,
                ),
                torch.zeros(
                    1, self.num_heads, self.block_size, self.head_dim,
                    dtype=self.dtype, device=self.device,
                ),
            )
            for _ in range(self.num_layers)
        ]

    def write_slot(
        self,
        block_id: int,
        layer: int,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one token's K/V into ``slot`` of ``block_id`` at ``layer``.

        ``key``/``value`` have shape ``(1, num_heads, 1, head_dim)``.
        """
        entry = self._blocks.get(block_id)
        if entry is None:
            entry = self._materialize()
            self._blocks[block_id] = entry
        k, v = entry[layer]
        k[:, :, slot : slot + 1, :] = key
        v[:, :, slot : slot + 1, :] = value
        if layer == 0:
            prev = self._slots.get(block_id, 0)
            if slot + 1 > prev:
                self._slots[block_id] = slot + 1
                self._tokens_stored += slot + 1 - prev

    def read_block(self, block_id: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self._blocks.get(block_id)
        if entry is None:
            entry = self._materialize()
            self._blocks[block_id] = entry
        return entry[layer]


class PagedSequence:
    """One sequence's logical KV state: block table + logical token count."""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_table: list[int] = []
        self.num_tokens = 0

    def ensure_slot(self) -> None:
        """Allocate a new physical block when the current one is full."""
        if self.num_tokens % self.allocator.block_size == 0:
            self.block_table.append(self.allocator.allocate())

    def append_kv(self, per_layer_kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Append one token's K/V (per-layer list of ``(key, value)``)."""
        self.ensure_slot()
        block_id = self.block_table[-1]
        slot = self.num_tokens % self.allocator.block_size
        for layer, (key, value) in enumerate(per_layer_kv):
            self.allocator.write_slot(block_id, layer, slot, key, value)
        self.num_tokens += 1

    def free(self) -> None:
        for block_id in self.block_table:
            self.allocator.free(block_id)
        self.block_table = []
        self.num_tokens = 0


def store_prefill(
    allocator: BlockAllocator,
    past_key_values: tuple,
    num_tokens: int,
) -> PagedSequence:
    """Chunk a prefill's contiguous KV cache into blocks for a new sequence.

    ``past_key_values`` is the legacy per-layer ``(key, value)`` tuple with
    shape ``(1, num_heads, num_tokens, head_dim)`` per tensor.
    """
    seq = PagedSequence(allocator)
    try:
        for pos in range(num_tokens):
            per_layer = [
                (key[:, :, pos : pos + 1, :], value[:, :, pos : pos + 1, :])
                for (key, value) in past_key_values
            ]
            seq.append_kv(per_layer)
    except Exception:
        seq.free()
        raise
    return seq


def gather_sequence(
    seq: PagedSequence,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Gather a sequence's blocks back into contiguous per-layer ``(key, value)``.

    Returns ``(layers, num_tokens)`` with each tensor shaped
    ``(1, num_heads, num_tokens, head_dim)``.
    """
    allocator = seq.allocator
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(allocator.num_layers):
        keys, values = [], []
        for block_id in seq.block_table:
            key, value = allocator.read_block(block_id, layer)
            keys.append(key)
            values.append(value)
        key = torch.cat(keys, dim=2)[:, :, : seq.num_tokens, :]
        value = torch.cat(values, dim=2)[:, :, : seq.num_tokens, :]
        layers.append((key, value))
    return layers, seq.num_tokens

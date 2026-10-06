import pytest
import torch

from app.paged import (
    BlockAllocator,
    PagedSequence,
    gather_sequence,
    store_prefill,
)


def make_allocator(num_blocks=8, block_size=4, num_layers=2):
    return BlockAllocator(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def fake_past_key_values(num_layers=2, seq_len=10):
    """Legacy-format cache with deterministic values for round-trip checks."""
    out = []
    for layer in range(num_layers):
        base = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
        key = base.expand(1, 2, seq_len, 8) + layer * 1000.0
        value = base.expand(1, 2, seq_len, 8) - layer * 1000.0
        out.append((key, value))
    return tuple(out)


def test_allocate_free_reuse_is_lifo():
    alloc = make_allocator()
    a = alloc.allocate()
    b = alloc.allocate()
    assert alloc.num_used == 2
    alloc.free(a)
    assert alloc.num_used == 1
    # LIFO: the just-freed block comes back first (cache warmth).
    assert alloc.allocate() == a
    assert alloc.num_used == 2
    alloc.free(a)
    alloc.free(b)
    assert alloc.num_used == 0
    assert alloc.num_free == 8


def test_pool_exhaustion_raises():
    alloc = make_allocator(num_blocks=2)
    alloc.allocate()
    alloc.allocate()
    with pytest.raises(RuntimeError, match="exhausted"):
        alloc.allocate()


def test_sequence_block_table_growth():
    alloc = make_allocator(num_blocks=8, block_size=4)
    seq = PagedSequence(alloc)
    pkv = fake_past_key_values(seq_len=10)
    stored = store_prefill(alloc, pkv, 10)
    assert stored.num_tokens == 10
    # 10 tokens at block_size 4 -> 3 physical blocks.
    assert len(stored.block_table) == 3
    assert alloc.num_used == 3
    # Appending fills the partial block first, then allocates.
    layers, n = gather_sequence(stored)
    assert n == 10
    knew = torch.full((1, 2, 1, 8), 7.0)
    vnew = torch.full((1, 2, 1, 8), -7.0)
    stored.append_kv([(knew, vnew), (knew, vnew)])
    assert stored.num_tokens == 11
    assert len(stored.block_table) == 3  # slot 2 of the third block
    for _ in range(5):
        stored.append_kv([(knew, vnew), (knew, vnew)])
    assert stored.num_tokens == 16
    assert len(stored.block_table) == 4  # new block at the 16-token boundary


def test_gather_roundtrip_matches_legacy_cache():
    alloc = make_allocator(num_blocks=8, block_size=4)
    pkv = fake_past_key_values(seq_len=10)
    seq = store_prefill(alloc, pkv, 10)
    layers, n = gather_sequence(seq)
    assert n == 10
    for (gathered_k, gathered_v), (orig_k, orig_v) in zip(layers, pkv):
        assert gathered_k.shape == orig_k.shape == (1, 2, 10, 8)
        assert torch.equal(gathered_k, orig_k)
        assert torch.equal(gathered_v, orig_v)


def test_free_reclaims_all_blocks():
    alloc = make_allocator(num_blocks=8, block_size=4)
    seq = store_prefill(alloc, fake_past_key_values(seq_len=10), 10)
    assert alloc.num_used == 3
    seq.free()
    assert alloc.num_used == 0
    assert alloc.num_free == 8
    assert alloc.fragmentation() == 0.0


def test_fragmentation_accounts_for_partial_blocks():
    alloc = make_allocator(num_blocks=8, block_size=4)
    store_prefill(alloc, fake_past_key_values(seq_len=10), 10)
    # 3 blocks x 4 slots = 12 slots, 10 used -> 2/12 fragmented.
    assert alloc.fragmentation() == pytest.approx(2 / 12)
    store_prefill(alloc, fake_past_key_values(seq_len=4), 4)
    # 4 blocks x 4 slots = 16 slots, 14 used -> 2/16 fragmented.
    assert alloc.fragmentation() == pytest.approx(2 / 16)


def test_prefill_failure_reclaims_blocks():
    alloc = make_allocator(num_blocks=8, block_size=4)
    calls = 0
    orig_allocate = alloc.allocate

    def flaky_allocate():
        nonlocal calls
        calls += 1
        if calls > 2:
            raise RuntimeError("boom")
        return orig_allocate()

    alloc.allocate = flaky_allocate
    with pytest.raises(RuntimeError, match="boom"):
        store_prefill(alloc, fake_past_key_values(seq_len=10), 10)
    # The two blocks allocated before the failure must be reclaimed.
    assert alloc.num_used == 0
    assert alloc.num_free == 8

"""Benchmark weight-only INT8 quantization on distilgpt2.

Compares three configurations on identical inputs:
  1. fp32            - baseline, no quantization
  2. int8-per-tensor - one scale per weight matrix
  3. int8            - one scale per output channel (recommended)

Reports, for each:
  - weight memory (MiB)
  - perplexity on a fixed held-out passage (lower is better)
  - mean milliseconds per generated token (greedy decode, CPU)
  - token-match rate vs the fp32 greedy output (quality proxy)

All inputs are fixed; rerunning reproduces the table. No network access
beyond the one-time model download.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, ".")
from app.quant import model_weight_bytes, quantize_model_

# Fixed evaluation passage (original prose, ~350 words). Perplexity is
# computed over this exact text for every configuration.
EVAL_TEXT = """The harbor woke before the town did. Gulls argued over the first
catch while the fishing boats, painted in fading blues and reds, rocked
gently against the wooden pilings. Mara had walked this pier every morning
for eleven years, and she knew each plank by the sound it made under her
boots. The third one from the bait shop creaked. The seventh was silent,
replaced last spring after the storm.

She carried her father's old thermos, dented on one side from the winter
it slid across the deck of the Marigold. The coffee inside was too strong,
the way he had always made it, and she had never learned to make it any
other way. Some habits are not habits at all but small acts of remembrance,
repeated until they feel like the morning itself.

The tide charts said the water would be low by nine. That gave her three
hours to check the crab pots along the eastern channel before the mud flats
turned the return trip into a careful negotiation. She had marked the pots
with orange buoys, though two had drifted in the night wind and would need
to be found by the shape of the water rather than by sight.

A seal surfaced near the breakwater, watched her for a moment with dark
unhurried eyes, and sank again without a sound. Mara smiled despite
herself. The old fishermen said seals were the harbor keeping an eye on
its own, and after eleven years she had stopped dismissing the idea
entirely. There were worse beliefs to carry into a morning.

By the time the sun cleared the cannery roof, she had the first pot aboard
and was working the line with the easy rhythm of long practice. The catch
was modest but clean, and the crabs clicked against the sorting tray in a
way that sounded, to her, like small applause."""

# Fixed generation prompts for the latency / token-match measurements.
PROMPTS = [
    "The harbor woke before the town did.",
    "She carried her father's old thermos,",
    "The tide charts said the water would be",
]


@torch.inference_mode()
def perplexity(model, tokenizer, text: str) -> float:
    ids = tokenizer.encode(text, return_tensors="pt")[0]
    # distilgpt2 context is 1024; our passage is shorter, single pass is fine.
    assert len(ids) < 1024, "eval passage exceeds model context"
    logits = model(ids.unsqueeze(0)).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    # NLL of each token given all previous tokens.
    nll = -log_probs[:-1].gather(1, ids[1:].unsqueeze(1)).squeeze(1)
    return float(torch.exp(nll.mean()))


@torch.inference_mode()
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int):
    ids = tokenizer.encode(prompt, return_tensors="pt")
    start = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start
    new_tokens = out.shape[1] - ids.shape[1]
    return out[0].tolist(), elapsed / max(new_tokens, 1)


# Fixed probe sentence for outlier-channel detection (LLM.int8() style).
# Tokenized once per config; the probe runs on the fp32 model before any
# quantization, so every config sees identical probe inputs.
PROBE_TEXT = ("The harbor woke before the town did. Gulls argued over the "
              "first catch while the fishing boats rocked gently. " * 4)


def build_config(name: str):
    """Load distilgpt2 and optionally quantize. Returns (model, tokenizer)."""
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    probe_ids = torch.tensor([tokenizer.encode(PROBE_TEXT)[:128]])
    if name == "int8":
        quantize_model_(model, per_channel=True, outlier_probe_ids=probe_ids)
    elif name == "int8-per-tensor":
        quantize_model_(model, per_channel=False, outlier_probe_ids=probe_ids)
    elif name != "fp32":
        raise ValueError(f"unknown config {name}")
    return model, tokenizer


def main(args):
    configs = ["fp32", "int8-per-tensor", "int8"]
    results = {}
    ref_tokens = None
    for name in configs:
        print(f"--- {name} ---", flush=True)
        model, tokenizer = build_config(name)
        size_mib = model_weight_bytes(model) / 2**20
        ppl = perplexity(model, tokenizer, EVAL_TEXT)
        latencies, matches, all_tokens = [], [], []
        for prompt in PROMPTS:
            tokens, ms_per_token = generate_greedy(
                model, tokenizer, prompt, args.max_new_tokens)
            latencies.append(ms_per_token * 1000)
            all_tokens.append(tokens)
            if ref_tokens is not None:
                ref = ref_tokens[PROMPTS.index(prompt)]
                matches.append(sum(a == b for a, b in zip(tokens, ref)) / len(ref))
        results[name] = {
            "size_mib": size_mib,
            "perplexity": ppl,
            "ms_per_token": sum(latencies) / len(latencies),
            "token_match": (sum(matches) / len(matches) if matches else 1.0),
        }
        if name == "fp32":
            ref_tokens = all_tokens
        del model

    print("\n=== INT8 quantization benchmark (distilgpt2, CPU, greedy) ===")
    print(f"{'config':<16}{'MiB':>8}{'ppl':>10}{'ms/tok':>10}{'tok-match':>10}")
    for name in configs:
        r = results[name]
        print(f"{name:<16}{r['size_mib']:>8.1f}{r['perplexity']:>10.2f}"
              f"{r['ms_per_token']:>10.1f}{r['token_match']:>10.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=32)
    main(parser.parse_args())

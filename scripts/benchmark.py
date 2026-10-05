from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * p / 100
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    frac = index - lo
    return values[lo] * (1 - frac) + values[hi] * frac


async def one_request(client, url, payload):
    start = time.perf_counter()
    response = await client.post(url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    elapsed = time.perf_counter() - start
    return {
        "latency": elapsed,
        "tokens": result["output_tokens"],
        "ttft": (result["time_to_first_token_ms"] or elapsed * 1000) / 1000,
    }


async def run_concurrent(base_url, n, concurrency, payload):
    semaphore = asyncio.Semaphore(concurrency)
    url = f"{base_url}/generate"

    async with httpx.AsyncClient() as client:
        async def wrapped():
            async with semaphore:
                return await one_request(client, url, payload)

        start = time.perf_counter()
        results = await asyncio.gather(*(wrapped() for _ in range(n)))
        elapsed = time.perf_counter() - start

    return results, elapsed


async def run_sequential(base_url, n, payload):
    url = f"{base_url}/generate"
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        results = []
        for _ in range(n):
            results.append(await one_request(client, url, payload))
        elapsed = time.perf_counter() - start
    return results, elapsed


def summarize(name, results, wall):
    total_tokens = sum(r["tokens"] for r in results)
    latencies = [r["latency"] * 1000 for r in results]
    ttfts = [r["ttft"] * 1000 for r in results]

    print(f"\n=== {name} ===")
    print(f"requests/sec:       {len(results) / wall:.2f}")
    print(f"tokens/sec:         {total_tokens / wall:.2f}")
    print(f"mean latency (ms):  {statistics.mean(latencies):.2f}")
    print(f"p50 latency (ms):   {percentile(latencies, 50):.2f}")
    print(f"p95 latency (ms):   {percentile(latencies, 95):.2f}")
    print(f"mean TTFT (ms):     {statistics.mean(ttfts):.2f}")

    return total_tokens / wall


async def main(args):
    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "top_k": 1,
    }

    print(f"Benchmarking {args.requests} requests, concurrency={args.concurrency}")
    print("Warm-up request...")
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{args.url}/generate",
            json={**payload, "max_new_tokens": 4},
            timeout=120,
        )
        r.raise_for_status()

    sequential_results, sequential_wall = await run_sequential(
        args.url, args.requests, payload
    )
    sequential_tps = summarize(
        "Sequential baseline", sequential_results, sequential_wall
    )

    concurrent_results, concurrent_wall = await run_concurrent(
        args.url, args.requests, args.concurrency, payload
    )
    concurrent_tps = summarize(
        "Continuous batching", concurrent_results, concurrent_wall
    )

    print("\n=== Improvement ===")
    print(f"token throughput:   {concurrent_tps / sequential_tps:.2f}x")
    print(f"wall-clock speedup: {sequential_wall / concurrent_wall:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Industrial and systems engineering improves",
    )
    asyncio.run(main(parser.parse_args()))

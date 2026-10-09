# From-Scratch LLM Inference Server with Continuous Batching

A master's-level systems project implementing an HTTP LLM inference server with:

- **Continuous batching**: requests arriving at different times share token-generation forward passes.
- **KV-cache management**: each active sequence keeps its own transformer attention cache.
- **Paged KV cache** (optional): vLLM-style fixed-size block allocation with per-sequence block tables, reclamation, and fragmentation metrics.
- **INT8 weight quantization** (optional): from-scratch symmetric quantization with integer GEMM, per-channel scales, and LLM.int8()-style outlier handling.
- **Prefill + decode phases**: prompt processing is separated from autoregressive generation.
- **Bounded scheduler**: max batch size, queue depth, request timeout, and generation limits.
- **Prometheus metrics**: request latency, time-to-first-token, tokens/sec, queue time, batch size, and KV-cache occupancy.
- **Streaming HTTP API** using Server-Sent Events (SSE).
- **Sequential baseline** for controlled benchmarking.
- **Dockerized execution**.
- Automated unit/integration tests.

The implementation uses PyTorch + Hugging Face Transformers for model execution, while the **serving/scheduling layer is implemented from scratch**.

> The goal is not to reproduce vLLM. The goal is to build a compact inference runtime that exposes the same core systems ideas clearly enough to benchmark and reason about.

---

## Architecture

```text
                     HTTP clients
                          |
                    FastAPI / SSE
                          |
                    Request Queue
                          |
                 +--------v---------+
                 | Continuous Batch |
                 |     Scheduler    |
                 +--------+---------+
                          |
               +----------+----------+
               |                     |
             PREFILL                DECODE
               |                     |
               +----------+----------+
                          |
                 ModelRunner / HF
                          |
                    KV cache per
                     sequence
                          |
                     Token output
                          |
                    SSE / JSON
```

### Continuous batching

A naïve server does:

```text
request A -> generate all tokens -> request B -> generate all tokens -> ...
```

This creates head-of-line blocking.

The scheduler instead operates at the token-generation step:

```text
step 1: [A, B, C]
step 2: [A, B, C]
step 3: [A, B]
step 4: [A, B, D]   <- D joined after C completed
step 5: [A, D]
...
```

The active set can therefore change between decode steps.

### KV cache

For autoregressive generation, previous attention keys and values do not need to be recomputed.

For each active request:

```text
request_id
   |
   +-- input_ids
   +-- generated_ids
   +-- past_key_values
   +-- attention position
   +-- sampling parameters
```

The cache belongs to the sequence, not to the HTTP connection.

### Paged KV cache

Set `PAGED_KV=1` to replace per-sequence contiguous caches with vLLM-style
block management (`app/paged.py`):

```text
physical pool: [block 0][block 1][block 2]...[block N]   (fixed size, e.g. 16 tokens)
sequence A block table: [7, 23]        -> tokens 0..31 live in blocks 7, 23
sequence B block table: [3]            -> tokens 0..15 live in block 3
```

- The `BlockAllocator` owns every physical block and reclaims them when a
  sequence finishes (including on prefill failure).
- Each sequence keeps a block table plus a logical token count; new blocks
  are allocated only when the current one is full, so sequences of different
  lengths no longer waste a full contiguous reservation each.
- New metrics: `inference_kv_blocks_total`, `inference_kv_blocks_used`,
  and `inference_kv_block_fragmentation_ratio` (share of allocated block
  capacity holding no token).

Honest scope note: this is the paged *management* layer. Before each forward
pass a sequence's blocks are gathered into one contiguous tensor for the HF
model call, and the new token's K/V is scattered back afterwards. A custom
block-sparse attention kernel would remove the gather step; the allocation,
reclamation, and fragmentation behavior implemented here is identical either
way. Equivalence with the legacy path is verified token-for-token
(`tests/test_paged.py`, plus a model-level check).

### INT8 weight quantization

Set `QUANTIZE=int8` to quantize the model after loading (`app/quant.py`):

- **Weights**: symmetric INT8, one scale per output channel. Every `nn.Linear`
  and HF `Conv1D` projection is replaced by an `Int8Linear` holding int8
  weights plus fp32 scales.
- **Compute**: activations are quantized per token at runtime and the matmul
  runs as integer GEMM (`torch._int_mm`, int8 x int8 accumulated in int32),
  descaled once afterwards. No dequantize-then-matmul: the integer path is
  the real thing, not quantization theater.
- **Tied embedding/head**: GPT-2 ties `lm_head` to the token embedding. The
  shared `[vocab, dim]` tensor is quantized once (per row); the embedding
  becomes an `Int8Embedding` (int8 gather + descale) and the head reuses the
  transposed int8 data. Nothing is forked or double-counted.
- **Activation outliers**: the final hidden state carries persistent outlier
  channels (3 of 768 channels reach magnitudes 40-150 while the median sits
  at 0.6) that a per-token activation scale would crush. Following
  LLM.int8(), those channels are detected with a short fixed probe at
  quantization time and computed in fp32 alongside the int8 GEMM.

Measured with `scripts/bench_quant.py` on distilgpt2 (CPU, greedy decode,
fixed prompts and evaluation passage; 2-vCPU AMD EPYC cloud VM):

| config | model size | perplexity | ms/token | greedy token-match vs fp32 |
|---|---|---|---|---|
| fp32 | 318.5 MiB | 89.66 | ~45 | 100% |
| int8 (per-channel) | 126.4 MiB | 90.88 (+1.4%) | ~16 | 53% |
| int8-per-tensor | 126.3 MiB | 105.39 (+17.5%) | ~19 | 32% |

The per-tensor row is the honest ablation: one scale per matrix cannot
represent layers with outlier channels, so perplexity degrades 17.5% --
which is exactly why the default is per-channel. Biases, layer norms and
the KV cache stay in float; fused INT8 kernels (which would widen the
latency win further) are out of scope.

---

## Project structure

```text
llm-inference-server/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── metrics.py
│   ├── models.py
│   ├── paged.py
│   ├── quant.py
│   ├── scheduler.py
│   └── server.py
├── scripts/
│   ├── benchmark.py
│   ├── bench_quant.py
│   └── smoke_test.py
├── tests/
│   ├── test_scheduler.py
│   ├── test_paged.py
│   ├── test_quant.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
├── .dockerignore
├── .gitignore
├── prometheus.yml
└── README.md
```

---

## Quick start

### Option A — Docker

```bash
docker compose up --build
```

The server starts at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Metrics:

```text
http://localhost:8000/metrics
```

Prometheus:

```text
http://localhost:9090
```

### Option B — Local Python

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Start:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The first request downloads the configured model if it is not already cached.

---

## API

### Generate

```http
POST /generate
Content-Type: application/json
```

Example:

```json
{
  "prompt": "Industrial engineering improves",
  "max_new_tokens": 32,
  "temperature": 0.7,
  "top_k": 50
}
```

Response:

```json
{
  "request_id": "...",
  "text": "...",
  "input_tokens": 4,
  "output_tokens": 32,
  "latency_ms": 1234.5,
  "time_to_first_token_ms": 350.1
}
```

### Streaming

```http
POST /generate/stream
```

The endpoint returns SSE events:

```text
event: token
data: {"request_id":"...","token":" improves"}

event: token
data: {"request_id":"...","token":" productivity"}

event: done
data: {"request_id":"...","output_tokens":12}
```

---

## Configuration

Environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_NAME` | `distilgpt2` | Hugging Face model |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `MAX_BATCH_SIZE` | `8` | Maximum active sequences per decode step |
| `MAX_QUEUE_SIZE` | `64` | Waiting request capacity |
| `MAX_NEW_TOKENS` | `128` | Server-side generation cap |
| `REQUEST_TIMEOUT_S` | `60` | Request timeout |
| `MODEL_DTYPE` | `float32` | `float32`, `float16`, or `bfloat16` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PAGED_KV` | `0` | `1` enables the paged KV-cache block manager |
| `KV_BLOCK_SIZE` | `16` | Tokens per physical KV block (paged mode) |
| `KV_NUM_BLOCKS` | `512` | Physical KV blocks in the pool (paged mode) |
| `QUANTIZE` | `none` | `int8` (per-channel) or `int8-per-tensor` weight quantization |

For a CPU-only laptop:

```bash
DEVICE=cpu MODEL_NAME=distilgpt2 uvicorn app.main:app --host 0.0.0.0
```

For a CUDA machine:

```bash
DEVICE=cuda MODEL_DTYPE=float16 uvicorn app.main:app --host 0.0.0.0
```

---

## Benchmark

Run the server first.

Then:

```bash
python scripts/benchmark.py --url http://localhost:8000 --requests 32 --concurrency 8
```

The benchmark compares:

1. **Sequential baseline** — one complete generation at a time.
2. **Continuous batching** — requests share decode steps.

It reports:

- requests/sec
- generated tokens/sec
- mean latency
- p50 latency
- p95 latency
- time-to-first-token
- throughput improvement

Example output:

```text
=== Sequential baseline ===
requests/sec:          0.82
tokens/sec:           21.4
mean latency (ms):  1187.2
p50 latency (ms):   1120.1
p95 latency (ms):   1512.7

=== Continuous batching ===
requests/sec:          2.61
tokens/sec:           68.3
mean latency (ms):   711.4
p50 latency (ms):    690.2
p95 latency (ms):    904.8

=== Improvement ===
throughput: 3.18x
```

**Do not claim these example numbers on your resume.** Run the benchmark on your actual hardware and report your measured result.

---

## Prometheus metrics

Important metrics:

```text
inference_requests_total
inference_request_latency_seconds
inference_time_to_first_token_seconds
inference_generated_tokens_total
inference_generation_tokens_per_second
inference_batch_size
inference_queue_time_seconds
inference_active_sequences
inference_kv_cache_sequences
```

Example PromQL:

```promql
rate(inference_generated_tokens_total[1m])
```

Average request latency:

```promql
rate(inference_request_latency_seconds_sum[5m])
/
rate(inference_request_latency_seconds_count[5m])
```

---

## Testing

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
pytest -q
```

The scheduler tests use a fake model runner, so they do not download a model.

The API integration tests use the same fake runner and therefore run quickly and deterministically.

---

## Design decisions

### Why Python?

The high-value systems work here is the scheduling policy, request lifecycle, cache ownership, batching semantics, metrics, and benchmarking. Python keeps the project focused on inference architecture instead of spending the entire project fighting memory safety and tensor bindings.

The model computation is still performed by PyTorch.

### Why token-level continuous batching?

A generation request can take dozens or hundreds of decode iterations. Holding a whole batch until every request completes wastes compute when requests have different output lengths.

By scheduling every decode iteration independently, completed sequences leave the active batch immediately and waiting sequences can enter.

### Why per-request KV cache?

A shared mutable cache would make sequence ownership and eviction difficult to reason about. Each active request owns its transformer cache, and the scheduler owns the lifecycle of that cache.

### What this project does NOT attempt

This is intentionally not a production replacement for vLLM/TensorRT-LLM.

It does not implement:

- block-sparse attention kernels (the paged *management* layer is implemented; the gather step before each forward pass is the remaining piece)
- CUDA kernels
- tensor parallelism
- speculative decoding
- distributed serving
- fused INT8 kernels (the integer GEMM path is implemented; a fused kernel
  would remove the per-forward activation-quantization overhead)
- FP8 / microscaling formats
- prefix caching
- GPU memory compaction

Those are natural master's/PhD-level extensions.

---

## Suggested master's-level extensions

If you want to push this beyond the baseline:

### 1. Block-sparse attention kernel

The paged block manager (`app/paged.py`) is implemented. The remaining step
is a custom kernel that reads K/V directly from scattered blocks, removing
the gather before each forward pass. Measure memory-bandwidth savings and
max-batch scaling with and without the kernel.

### 2. Admission control

Implement:

```text
queue wait budget
max prompt tokens
max active KV tokens
```

### 3. Scheduling policies

Compare:

- FIFO
- shortest-output-first
- aging
- token-budget scheduling

### 4. Prefix caching

Hash the prompt prefix and reuse its KV cache.

### 5. Quantization

Compare FP32 / FP16 / INT8 inference.

### 6. GPU profiling

Measure:

```text
GPU utilization
kernel launch overhead
memory bandwidth
batch-size scaling
```

---

## Resume bullet template

After measuring the actual result:

> Built a from-scratch Python/PyTorch LLM inference server implementing continuous batching and per-sequence KV-cache management; containerized the runtime and instrumented Prometheus metrics, achieving **X× higher token throughput** than sequential serving under an **N-request concurrent workload**.

That is much stronger than saying "built an LLM chatbot."

---

## Important reproducibility note

Performance depends heavily on:

- CPU/GPU model
- PyTorch version
- model weights
- batch size
- prompt length
- output length
- concurrency
- thermal throttling

Always report the benchmark configuration alongside the result.

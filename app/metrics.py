from prometheus_client import Counter, Gauge, Histogram


REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests.",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

TTFT = Histogram(
    "inference_time_to_first_token_seconds",
    "Time from admission to first generated token.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

QUEUE_TIME = Histogram(
    "inference_queue_time_seconds",
    "Time spent waiting before entering an active batch.",
    buckets=(0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

GENERATED_TOKENS = Counter(
    "inference_generated_tokens_total",
    "Total generated tokens.",
)

GENERATION_SECONDS = Counter(
    "inference_generation_seconds_total",
    "Total seconds spent generating.",
)

BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Decode batch size.",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

ACTIVE_SEQUENCES = Gauge(
    "inference_active_sequences",
    "Number of active generation sequences.",
)

KV_SEQUENCES = Gauge(
    "inference_kv_cache_sequences",
    "Number of sequences currently holding KV cache.",
)

KV_BLOCKS_TOTAL = Gauge(
    "inference_kv_blocks_total",
    "Total physical KV blocks in the pool (paged mode).",
)

KV_BLOCKS_USED = Gauge(
    "inference_kv_blocks_used",
    "Physical KV blocks currently allocated (paged mode).",
)

KV_BLOCK_FRAGMENTATION = Gauge(
    "inference_kv_block_fragmentation_ratio",
    "Share of allocated block capacity holding no token (paged mode).",
)

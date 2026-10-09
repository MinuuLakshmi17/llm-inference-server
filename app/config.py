from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "distilgpt2")
    device: str = os.getenv("DEVICE", "auto")
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "8"))
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "64"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
    model_dtype: str = os.getenv("MODEL_DTYPE", "float32")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    paged_kv: bool = os.getenv("PAGED_KV", "0") == "1"
    kv_block_size: int = int(os.getenv("KV_BLOCK_SIZE", "16"))
    kv_num_blocks: int = int(os.getenv("KV_NUM_BLOCKS", "512"))
    # Weight-only INT8 quantization: "none" | "int8" (per-channel) | "int8-per-tensor".
    quantize: str = os.getenv("QUANTIZE", "none")


settings = Settings()

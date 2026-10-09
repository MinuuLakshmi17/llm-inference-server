import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from app.quant import (
    Int8Linear,
    model_weight_bytes,
    quantize_activation_per_token,
    quantize_model_,
    quantize_symmetric,
)


def dequant(q, scale):
    return q.float() * scale


def test_symmetric_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    for per_channel in (True, False):
        q, scale = quantize_symmetric(w, per_channel=per_channel, channel_dim=0)
        assert q.dtype == torch.int8
        assert q.min() >= -128 and q.max() <= 127
        w_hat = dequant(q, scale)
        step = scale.amax() if per_channel else scale
        assert (w - w_hat).abs().max() <= step / 2 + 1e-6


def test_per_channel_beats_per_tensor_on_outliers():
    torch.manual_seed(1)
    w = torch.randn(8, 64) * 0.01
    w[3] *= 100.0  # one output channel holds outlier magnitudes
    err_t = (w - dequant(*quantize_symmetric(w, per_channel=False, channel_dim=0))).abs().mean()
    err_c = (w - dequant(*quantize_symmetric(w, per_channel=True, channel_dim=0))).abs().mean()
    assert err_c < err_t


def test_zero_weight_has_no_nan_or_inf():
    w = torch.zeros(16, 16)
    q, scale = quantize_symmetric(w, per_channel=True, channel_dim=0)
    assert torch.isfinite(scale).all()
    assert (dequant(q, scale) == 0).all()


def test_activation_quant_per_token_scales():
    torch.manual_seed(4)
    x = torch.randn(5, 32)
    x[0] *= 50.0  # one loud token must not crush the quiet ones
    q, scale = quantize_activation_per_token(x)
    assert q.dtype == torch.int8
    assert scale.shape == (5, 1)
    # The quiet tokens keep their own small scale, not the loud token's.
    assert scale[1] < scale[0] / 10


def test_int8_linear_matches_fp32():
    torch.manual_seed(2)
    linear = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    expected = linear(x)
    got = Int8Linear(linear, per_channel=True)(x)
    assert got.shape == expected.shape
    rel = (got - expected).abs().max() / expected.abs().max()
    assert rel < 0.05


def test_int8_conv1d_matches_fp32():
    torch.manual_seed(5)
    conv = Conv1D(16, 32)  # Conv1D(nf=out, nx=in), forward is x @ W + b
    x = torch.randn(4, 10, 32)
    expected = conv(x)
    got = Int8Linear(conv, per_channel=True)(x)
    assert got.shape == expected.shape
    rel = (got - expected).abs().max() / expected.abs().max()
    assert rel < 0.05


def test_int8_linear_keeps_bias():
    linear = nn.Linear(8, 4, bias=True)
    qlinear = Int8Linear(linear)
    assert qlinear.bias is not None
    assert torch.equal(qlinear.bias.data, linear.bias.data)
    assert qlinear.weight_q.dtype == torch.int8
    # Stored transposed to [in, out] for the integer GEMM.
    assert qlinear.weight_q.shape == (8, 4)


def test_quantize_model_replaces_linears_and_conv1d():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), Conv1D(16, 64))
    stats = quantize_model_(model, per_channel=True)
    assert stats.num_linear_layers == 2
    assert stats.num_skipped_tied == 0
    assert sum(isinstance(m, Int8Linear) for m in model.modules()) == 2
    assert not any(type(m) is nn.Linear for m in model.modules())
    assert not any(type(m) is Conv1D for m in model.modules())
    # Weight-only compression should sit just under 4x (fp32 scales cost).
    assert 3.5 < stats.compression_ratio < 4.0


def test_quantize_model_skip_names():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
    stats = quantize_model_(model, skip_names=("1",))
    assert stats.num_linear_layers == 1


def test_tied_embedding_and_head_share_one_quantized_tensor():
    torch.manual_seed(6)

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 16)
            self.head = nn.Linear(16, 32, bias=False)
            self.head.weight = self.emb.weight  # tied storage

        def forward(self, ids):
            return self.head(self.emb(ids))

    model = Tied()
    before = model_weight_bytes(model)
    probe = torch.tensor([[1, 2, 3, 4]])
    stats = quantize_model_(model, outlier_probe_ids=probe, outlier_threshold=1e9)
    assert stats.num_skipped_tied == 0
    assert stats.num_linear_layers == 1  # the head; the embedding rides along
    from app.quant import Int8Embedding
    assert isinstance(model.emb, Int8Embedding)
    assert isinstance(model.head, Int8Linear)
    # Absurdly high threshold -> no outliers found -> pure int8 path.
    assert len(model.head.outlier_idx) == 0
    # One int8 copy for the embedding + one transposed copy for the GEMM:
    # still far smaller than the two fp32 copies a fork would need.
    assert model_weight_bytes(model) < before
    # Functional check: embedding lookup then head still works.
    ids = torch.tensor([[1, 2, 3]])
    out = model(ids)
    assert out.shape == (1, 3, 32)
    assert torch.isfinite(out).all()


def test_outlier_side_path_matches_fp32():
    # Force outlier channels with a spiky input; the side path must rescue them.
    torch.manual_seed(9)
    linear = nn.Linear(32, 16, bias=False)
    qlinear = Int8Linear(linear, outlier_idx=torch.tensor([3, 7]))
    x = torch.randn(4, 32) * 0.5
    x[:, 3] *= 200.0
    x[:, 7] *= 150.0
    expected = linear(x)
    got = qlinear(x)
    rel = (got - expected).abs().max() / expected.abs().max()
    # Without the side path these outliers would be crushed; with it, close.
    assert rel < 0.05
    # Sanity: the same module *without* outlier handling is far worse.
    plain = Int8Linear(linear)
    bad = plain(x)
    rel_bad = (bad - expected).abs().max() / expected.abs().max()
    assert rel_bad > rel


def test_probe_outlier_channels_finds_spikes():
    torch.manual_seed(10)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8, bias=False)

        def forward(self, x):
            # Inject a spike on channel 5 of the fc input.
            h = torch.randn(x.shape[0], 16) * 0.1
            h[:, 5] = 50.0
            return self.fc(h)

    from app.quant import probe_outlier_channels
    model = Net()
    found = probe_outlier_channels(model, "fc", torch.tensor([[1, 2]]), threshold=6.0)
    assert 5 in found.tolist()


def test_tied_to_non_embedding_stays_fp32():
    torch.manual_seed(7)

    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)
            self.alias = nn.Parameter(self.proj.weight)  # tied, not embedding

        def forward(self, x):
            return self.proj(x)

    model = Weird()
    before = model_weight_bytes(model)
    stats = quantize_model_(model)
    assert stats.num_skipped_tied == 1
    assert stats.num_linear_layers == 0
    assert model_weight_bytes(model) == before
    assert type(model.proj) is nn.Linear


def test_int8_embedding_matches_fp32():
    torch.manual_seed(8)
    emb = nn.Embedding(64, 16)
    ids = torch.tensor([[1, 5, 63], [0, 2, 4]])
    expected = emb(ids)
    from app.quant import Int8Embedding
    got = Int8Embedding(emb)(ids)
    assert got.shape == expected.shape
    rel = (got - expected).abs().max() / expected.abs().max()
    assert rel < 0.02


def _tiny_gpt2():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128,
                        n_positions=64, n_ctx=64)
    torch.manual_seed(42)
    return GPT2LMHeadModel(config).eval()


def test_tiny_gpt2_quantized_generates_valid_tokens():
    model = _tiny_gpt2()
    stats = quantize_model_(model, per_channel=True)
    assert stats.num_linear_layers > 0
    # lm_head is tied to the token embedding: quantized once, shared.
    from app.quant import Int8Embedding
    assert isinstance(model.transformer.wte, Int8Embedding)
    assert isinstance(model.lm_head, Int8Linear)
    input_ids = torch.tensor([[10, 20, 30]])
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=5, do_sample=False,
                             pad_token_id=0)
    assert out.shape == (1, 8)
    assert out.min() >= 0 and out.max() < 128


def test_tiny_gpt2_quantized_logits_close_to_fp32():
    torch.manual_seed(42)
    fp32 = _tiny_gpt2()
    torch.manual_seed(42)
    int8 = _tiny_gpt2()
    quantize_model_(int8, per_channel=True)
    input_ids = torch.tensor([[10, 20, 30, 40]])
    with torch.inference_mode():
        l_fp32 = fp32(input_ids).logits
        l_int8 = int8(input_ids).logits
    diff = (l_fp32 - l_int8).abs()
    assert (diff.amax() / l_fp32.abs().amax()) < 0.15

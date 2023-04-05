import pytest
import numpy as np
import mindspore
from mindspore import Tensor
import torch
from transformers import AutoModel

from src.configuration_bloom import BloomConfig
from src.modeling_bloom import BloomModel


def test_bloom_forward():
    config = BloomConfig(seq_length=20, vocab_size=2000)
    model = BloomModel(config)

    input_ids = Tensor(np.random.randint(0, 2000, (1, 20)), mindspore.int32)
    input_mask = Tensor(np.ones((1, 20)), mindspore.float32)
    outputs = model(input_ids, input_mask)

def test_hf_bloom_bf16_small_testing():
    hf_bloom = AutoModel.from_pretrained('bigscience/bigscience-small-testing')
    # hf_bloom = AutoModel.from_pretrained('bigscience/bloom-560m')
    input_ids = torch.randint(0, 200, (1, 20))
    outputs = hf_bloom(input_ids)
    hf_bloom.to(torch.bfloat16)
    print(hf_bloom.config)

    outputs_bf16 = hf_bloom(input_ids)
    print(outputs_bf16.last_hidden_state.dtype)

    error_count_1e_minus_3 = ((outputs.last_hidden_state.detach().numpy() - \
          outputs_bf16.last_hidden_state.to(torch.float32).detach().numpy()) > 1e-3).sum()
    error_count_5e_minus_3 = ((outputs.last_hidden_state.detach().numpy() - \
          outputs_bf16.last_hidden_state.to(torch.float32).detach().numpy()) > 5e-3).sum()
    total = outputs.last_hidden_state.numel()
    print(total, error_count_1e_minus_3, error_count_5e_minus_3)
    assert np.allclose(outputs.last_hidden_state.detach().numpy(),
                       outputs_bf16.last_hidden_state.to(torch.float32).detach().numpy(),
                       1e-2, 1e-2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='LayerNorm not support fp16 on CPU')
def test_hf_bloom_fp16_560m():
    hf_bloom = AutoModel.from_pretrained('bigscience/bloom-560m')
    input_ids = torch.randint(0, 200, (1, 20))
    outputs = hf_bloom(input_ids)
    hf_bloom.to(torch.bfloat16)
    print(hf_bloom.config)

    outputs_bf16 = hf_bloom(input_ids)
    print(outputs_bf16.last_hidden_state.dtype)

    error_count_1e_minus_3 = ((outputs.last_hidden_state.detach().numpy() - \
          outputs_bf16.last_hidden_state.to(torch.float32).detach().numpy()) > 1e-3).sum()
    error_count_5e_minus_3 = ((outputs.last_hidden_state.detach().numpy() - \
          outputs_bf16.last_hidden_state.to(torch.float32).detach().numpy()) > 5e-3).sum()
    total = outputs.last_hidden_state.numel()
    print(total, error_count_1e_minus_3, error_count_5e_minus_3)
    assert np.allclose(outputs.last_hidden_state.detach().numpy(),
                       outputs_bf16.last_hidden_state.to(torch.float32).detach().numpy(),
                       1e-2, 1e-2)

import pytest
import numpy as np
import mindspore
from mindspore import Tensor
import torch
from transformers import AutoModel

from src.configuration_bloom import BloomConfig
from src.modeling_bloom import BloomModel
from src.mapping import hf_to_ms

from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='LayerNorm not support fp16 on CPU')
def test_hf_bloom_fp16_560m():
    hf_bloom = AutoModel.from_pretrained('bigscience/bloom-560m', torch_dtype=torch.float16)
    input_ids = torch.randint(0, 200, (1, 20)).cuda()
    hf_bloom.cuda()
    outputs_fp16 = hf_bloom(input_ids)
    print(outputs_fp16.last_hidden_state.dtype)

    hf_bloom.float()
    outputs = hf_bloom(input_ids)

    outputs_last_hidden_state = outputs.last_hidden_state.cpu().detach().numpy()
    outputs_fp16_last_hidden_state = outputs_fp16.last_hidden_state.to(torch.float32).cpu().detach().numpy()
    error_count_1e_minus_3 = ((outputs_last_hidden_state - \
          outputs_fp16_last_hidden_state) > 1e-3).sum()
    error_count_5e_minus_3 = ((outputs_last_hidden_state - \
          outputs_fp16_last_hidden_state) > 5e-3).sum()
    total = outputs.last_hidden_state.numel()
    print(total, error_count_1e_minus_3, error_count_5e_minus_3)


def test_bloom_hf_precision_small_test():
    hf_bloom = AutoModel.from_pretrained('bigscience/bigscience-small-testing')
    hf_bloom.eval()
    # convert hf ckpt to ms
    print(hf_bloom.config)
    hf_weights = hf_bloom.state_dict()

    ms_params = hf_to_ms(hf_weights, hf_bloom.config)
    ms_config = BloomConfig(hidden_size=64, num_heads=8, num_layers=2, seq_length=20)
    ms_bloom = BloomModel(ms_config)
    ms_bloom.set_train(False)

    not_loaded = mindspore.load_param_into_net(ms_bloom, ms_params)
    print(not_loaded)

    input_ids = np.random.randint(0, 2000, (1, 20))
    input_mask = np.ones((1, 20))

    input_ids_ms = Tensor(input_ids, mindspore.int32)
    input_mask_ms = Tensor(input_mask, mindspore.float32)

    input_ids_pt = torch.tensor(input_ids, dtype=torch.int32)
    input_mask_pt = torch.tensor(input_mask, dtype=torch.float32)

    outputs_ms = ms_bloom(input_ids_ms, input_mask_ms)
    outputs_hf = hf_bloom(input_ids_pt, attention_mask=input_mask_pt)

    hidden_hf = outputs_hf.last_hidden_state.detach().numpy()
    hidden_ms = outputs_ms[0].asnumpy()

    total = outputs_hf.last_hidden_state.numel()
    error_count_1e_minus_3 = ((hidden_hf - \
          hidden_ms) > 1e-3).sum()
    error_count_5e_minus_3 = ((hidden_hf - \
          hidden_ms) > 5e-3).sum()
    print(total, error_count_1e_minus_3, error_count_5e_minus_3)
    print(hidden_hf - hidden_ms)

    assert np.allclose(outputs_ms[0].asnumpy(),
                       outputs_hf.last_hidden_state.detach().numpy(),
                       5e-3, 5e-3
                       )

def test_bloom_hf_precision_560m():
    hf_bloom = AutoModel.from_pretrained('bigscience/bloom-560m')
    hf_bloom.eval()
    # convert hf ckpt to ms
    print(hf_bloom.config)
    hf_weights = hf_bloom.state_dict()

    ms_params = hf_to_ms(hf_weights, hf_bloom.config)
    ms_config = BloomConfig(hidden_size=1024, num_heads=16, num_layers=24, seq_length=20)
    ms_bloom = BloomModel(ms_config)
    ms_bloom.set_train(False)

    not_loaded = mindspore.load_param_into_net(ms_bloom, ms_params)
    print(not_loaded)

    input_ids = np.random.randint(0, 2000, (1, 20))
    input_mask = np.ones((1, 20))

    input_ids_ms = Tensor(input_ids, mindspore.int32)
    input_mask_ms = Tensor(input_mask, mindspore.float32)

    input_ids_pt = torch.tensor(input_ids, dtype=torch.int32)
    input_mask_pt = torch.tensor(input_mask, dtype=torch.float32)

    outputs_ms = ms_bloom(input_ids_ms, input_mask_ms)

    outputs_hf = hf_bloom(input_ids_pt, attention_mask=input_mask_pt)

    hidden_hf = outputs_hf.last_hidden_state.detach().to(torch.float32).numpy()
    hidden_ms = outputs_ms[0].asnumpy()

    total = outputs_hf.last_hidden_state.numel()
    error_count_1e_minus_3 = ((hidden_hf - \
          hidden_ms) > 1e-3).sum()
    error_count_5e_minus_3 = ((hidden_hf - \
          hidden_ms) > 5e-3).sum()
    print(total, error_count_1e_minus_3, error_count_5e_minus_3)

    error = hidden_hf - hidden_ms
    print(error)
    print(error.shape)
    # np.save('error.npy', hidden_hf - hidden_ms)
    assert np.allclose(outputs_ms[0].asnumpy(),
                       outputs_hf.last_hidden_state.detach().to(torch.float32).numpy(),
                       1e-3, 1e-3
                       )
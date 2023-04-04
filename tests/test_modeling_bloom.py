import pytest
import numpy as np
import mindspore
from mindspore import Tensor

from src.configuration_bloom import BloomConfig
from src.modeling_bloom import BloomModel


def test_bloom_forward():
    config = BloomConfig(seq_length=20, vocab_size=2000)
    model = BloomModel(config)

    input_ids = Tensor(np.random.randint(0, 2000, (1, 20)), mindspore.int32)
    input_mask = Tensor(np.ones((1, 20)), mindspore.float32)
    outputs = model(input_ids, input_mask)

def test_hf_bloom():
    import torch
    from transformers import AutoModel
    hf_bloom = AutoModel.from_pretrained('bigscience/bigscience-small-testing')
    print(hf_bloom.config)
    input_ids = torch.randint(0, 200, (1, 20))

    outputs = hf_bloom(input_ids)
    print(outputs.last_hidden_state.dtype)

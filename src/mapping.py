import mindspore
from mindspore import Tensor, Parameter
from transformers import AutoModel
from src.configuration_bloom import BloomConfig
from src.modeling_bloom import BloomModel
import re

def layer_name_mapping(key):
    """Convert huggingface PP weights mapping in MindSpore.
    
    return: split, new_name
    """
    prefix = ''
    if 'transformer' in key:
        prefix = 'transformer.'
        key = key.replace('transformer.', '')
    # Handle first and last layers
    layer_rename_map = {
        "word_embeddings.weight": "embedding.word_embeddings.embedding_table",
        "word_embeddings_layernorm.weight": "embedding.norm.gamma",
        "word_embeddings_layernorm.bias": "embedding.norm.beta",
        "ln_f.weight": "ln_f.gamma",
        "ln_f.bias": "ln_f.beta",
        "input_layernorm.weight": "layernorm1.gamma",
        "input_layernorm.bias": "layernorm1.beta",
        "self_attention.query_key_value.weight": "attention.dense{}.weight",
        "self_attention.query_key_value.bias": "attention.dense{}.bias",
        "self_attention.dense.weight": "attention.projection.weight",
        "self_attention.dense.bias": "attention.projection.bias",
        "post_attention_layernorm.weight": "layernorm2.gamma",
        "post_attention_layernorm.bias": "layernorm2.beta",
        "mlp.dense_h_to_4h.weight": "output.mapping.weight",
        "mlp.dense_h_to_4h.bias": "output.mapping.bias",
        "mlp.dense_4h_to_h.weight": "output.projection.weight",
        "mlp.dense_4h_to_h.bias": "output.projection.bias",
        "lm_head.weight": "head.weight",
        "lm_head.bias": "head.bias",
    }

    split = False
    if key in layer_rename_map:
        return split, prefix + layer_rename_map[key]

    # Handle transformer blocks
    match = re.match(r'^\w*\.(\d+)\.(\w+\.\w+\.\w+|\w+\.\w+)$', key)
    layer_number = int(match.group(1))
    text = match.group(2)
    if "self_attention.query_key_value" in key:
        split = True
    return split, f"{prefix}blocks.{layer_number}." + layer_rename_map[text]

def hf_to_ms(hf_weights, config):
    ms_params = {}
    for k, v in hf_weights.items():
        print(k, v.shape, v.dtype)
        split, new_name = layer_name_mapping(k)
        if split:
            if 'weight' in new_name:
                v = v.reshape(config.n_head, 3, config.hidden_size // config.n_head, v.shape[-1])
                v_list = v.tensor_split(3, dim=1)
                for i in range(1, 4):
                    tmp_name = new_name.format(i)
                    print(v_list[i-1].shape)
                    tmp_tensor = Tensor(v_list[i-1].reshape(-1, v_list[i-1].shape[-1]).numpy(), mindspore.float32)
                    ms_params[tmp_name] = Parameter(tmp_tensor, name=tmp_name)
            else:
                v = v.reshape(config.n_head, 3, config.hidden_size // config.n_head)
                v_list = v.tensor_split(3, dim=1)
                for i in range(1, 4):
                    tmp_name = new_name.format(i)
                    print(v_list[i-1].shape)
                    tmp_tensor = Tensor(v_list[i-1].reshape(-1).numpy(), mindspore.float32)
                    # if 'weight' in new_name:
                    #     tmp_tensor = tmp_tensor.swapaxes(0, 1)
                    ms_params[tmp_name] = Parameter(tmp_tensor, name=tmp_name)
        else:
            if ('projection' in new_name or 'mapping' in new_name) and 'weight' in new_name:
                new_tensor = Tensor(v.transpose(0, 1).numpy(), mindspore.float32)
            else:
                new_tensor = Tensor(v.numpy(), mindspore.float32)
            ms_params[new_name] = Parameter(new_tensor, name=new_name)

    return ms_params



if __name__ == '__main__':
    hf_bloom = AutoModel.from_pretrained('bigscience/bigscience-small-testing')

    # convert hf ckpt to ms
    print(hf_bloom.config)
    hf_weights = hf_bloom.state_dict()

    ms_params = hf_to_ms(hf_weights, hf_bloom.config)
    ms_config = BloomConfig(hidden_size=64, num_heads=8, num_layers=2, seq_length=20)
    ms_bloom = BloomModel(ms_config)


    for k, v in ms_bloom.parameters_and_names():
        print(k, v.shape, v.dtype)

    not_loaded = mindspore.load_param_into_net(ms_bloom, ms_params)
    pass

import torch
import mindspore
from mindspore import Tensor, Parameter
from transformers import AutoConfig
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

def hf_to_ms(hf_weights, config, ms_dtype=mindspore.float32, for_save=False):
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
                    tmp_tensor = Tensor(v_list[i-1].reshape(-1, v_list[i-1].shape[-1]).float().numpy(), ms_dtype)
                    ms_params[tmp_name] = Parameter(tmp_tensor, name=tmp_name)
            else:
                v = v.reshape(config.n_head, 3, config.hidden_size // config.n_head)
                v_list = v.tensor_split(3, dim=1)
                for i in range(1, 4):
                    tmp_name = new_name.format(i)
                    print(v_list[i-1].shape)
                    tmp_tensor = Tensor(v_list[i-1].reshape(-1).float().numpy(), ms_dtype)
                    # if 'weight' in new_name:
                    #     tmp_tensor = tmp_tensor.swapaxes(0, 1)
                    ms_params[tmp_name] = Parameter(tmp_tensor, name=tmp_name)
        else:
            if ('projection' in new_name or 'mapping' in new_name) and 'weight' in new_name:
                new_tensor = Tensor(v.transpose(0, 1).float().numpy(), ms_dtype)
            else:
                new_tensor = Tensor(v.float().numpy(), ms_dtype)
            ms_params[new_name] = Parameter(new_tensor, name=new_name)

    if for_save:
        return [{'name':k, 'data':v} for k,v in ms_params.items()]

    return ms_params

def process_hf_shard_files(file_list, config, save_dir=None, combine=False, ms_dtype=mindspore.float32):
    combine_params = []
    for file in file_list:
        pt_states = torch.load(file, map_location='cpu')
        ms_params = hf_to_ms(pt_states, config, ms_dtype, True)
        if combine:
            combine_params.extend(ms_params)
        else:
            save_file = save_dir + '/' + file.split('/')[-1] if save_dir else file + '.ckpt'
            mindspore.save_checkpoint(ms_params, save_file)

        del pt_states
        del ms_params

    if combine:
        path = save_dir + '/' + 'combine.ckpt' if save_dir else \
            '/'.join(file.split('/')[:-1]) + 'combine.ckpt'
        mindspore.save_checkpoint(combine_params, save_dir)

def hf_combined_to_ms(src_dir, dst_dir, ckpt_prefix, out_strategy):
    # src strategy is None since hf ckpt is not shard for mp or pp.
    mindspore.transform_checkpoints(src_dir, dst_dir,
                                    ckpt_prefix=ckpt_prefix,
                                    src_strategy_file=None,
                                    dst_strategy_file=out_strategy)

if __name__ == '__main__':
    config = AutoConfig.from_pretrained('bigscience/bigscience-small-testing')

    print(config)
    # convert hf ckpt to ms
    process_hf_shard_files(['pytorch_model.bin'], config, ms_dtype=mindspore.float16)
    process_hf_shard_files(['pytorch_model.bin'], config, combine=True)

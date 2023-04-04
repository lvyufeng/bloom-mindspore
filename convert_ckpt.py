from transformers import AutoModel
from src.configuration_bloom import BloomConfig
from src.modeling_bloom import BloomModel
import re

hf_bloom = AutoModel.from_pretrained('bigscience/bloom-560m')

print(hf_bloom.config)
hf_weights = hf_bloom.state_dict()

for k, v in hf_weights.items():
    print(k, v.shape, v.dtype)

ms_config = BloomConfig(hidden_size=64, num_heads=8, num_layers=2, seq_length=20)
ms_bloom = BloomModel(ms_config)


for k, v in ms_bloom.parameters_and_names():
    print(k, v.shape, v.dtype)

def layer_name_mapping(key, file):
    """Convert huggingface PP weights mapping in MindSpore."""
    # Handle first and last layers
    layer_rename_map = {
        "word_embeddings.weight": "embedding.word_embeddings.embedding_table",
        "word_embeddings_layernorm.weight": "embedding.norm.gamma",
        "word_embeddings_layernorm.bias": "embedding.norm.beta",
        "ln_f.weight": "ln_f.gamma",
        "ln_f.bias": "ln_f.beta",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]

    # Handle transformer blocks
    layer_number = int(re.match(r".*layer_(\d*).*", file)[1])
    layer_number -= 3
    return f"h.{layer_number}." + key


pass

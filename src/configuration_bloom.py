from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.base_config import BaseConfig

class BloomConfig(BaseConfig):
    """
    Bloom config class which defines the model size
    """

    def __init__(self,
                 dropout_prob: float = 0.1,
                 batch_size: int = None,
                 seq_length: int = 1024,
                 vocab_size: int = 250880,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 expand_ratio: int = 4,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 bos_token: int = 1,
                 eos_token: int = 2,
                 param_init_type: str = "float32",
                 layernorm_dtype: str = "float32",
                 softmax_dtype: str = "float32",
                 compute_dtype: str = "float16",
                 hidden_act: str = 'gelu', # mindspore.nn.GELU(approximate=True) eq bloomGELU
                 parallel_config: TransformerOpParallelConfig = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: MoEConfig = default_moe_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_dtype = convert_mstype(layernorm_dtype)
        self.softmax_dtype = convert_mstype(softmax_dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.hidden_act = hidden_act

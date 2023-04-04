from mindspore import nn, ops
from mindspore import Parameter
import mindspore.log as logger
from mindspore.common.initializer import initializer, TruncatedNormal
from mindformers.modules.transformer.transformer import default_embedding_parallel_config
from .layer_norm import LayerNorm

class VocabEmbedding(nn.Cell):
    def __init__(self, vocab_size, embedding_size, embed_layernorm=True, parallel_config=default_embedding_parallel_config,
                 param_init='normal'):
        super(VocabEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                            name='weight', parallel_optimizer=False)

        if parallel_config.vocab_emb_dp:
            self.gather = ops.Gather().shard(((1, 1), (parallel_config.data_parallel, 1)))
            logger.info(f"Using {parallel_config.data_parallel} data parallel for the embedding lookup.")
        else:
            if self.vocab_size % parallel_config.model_parallel != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of parallel_config.model_parallel {parallel_config.model_parallel}.")
            self.gather = ops.Gather().shard(((parallel_config.model_parallel, 1), (parallel_config.data_parallel, 1)))
            logger.info(f"Using {parallel_config.data_parallel} data parallel and {parallel_config.model_parallel} "
                        f"model parallel for the embedding lookup.")
        
        self.embed_layernorm = embed_layernorm
        if embed_layernorm:
            self.norm = LayerNorm((embedding_size,))

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        if self.embed_layernorm:
            output = self.norm(output)
        return output, self.embedding_table.value()

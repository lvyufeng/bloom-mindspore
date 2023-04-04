import copy
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.layers import LayerNorm, Dropout
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.transformer import AttentionMask, TransformerEncoder, VocabEmbedding
from mindformers.models.base_model import BaseModel
from mindformers.tools.logger import logger
from .configuration_bloom import BloomConfig


class BloomEmbeddingLayer(nn.Cell):
    r"""The Embedding Layer of GPT-2 network."""
    def __init__(self, config = None):
        super().__init__(auto_prefix=False)
        parallel_config = copy.deepcopy(config.parallel_config)
        embedding_mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        vocab_size = config.vocab_size
        if vocab_size % embedding_mp != 0:
            logger.warning("The vocab size of embedding layer is: %s, it is not divide by model_parallel: %s",
                           vocab_size, embedding_mp)
            logger.warning("Now, model_parallel will be changed: mp = 1")
            parallel_config.embedding_dp_mp_config.model_parallel = 1

        self.word_embeddings = VocabEmbedding(vocab_size=vocab_size,
                                              embedding_size=config.hidden_size,
                                              param_init=initializer(TruncatedNormal(config.initializer_range),
                                                                     [vocab_size, config.hidden_size],
                                                                     dtype=mstype.float32),
                                              parallel_config=parallel_config.embedding_dp_mp_config)
        new_parallel_config = copy.deepcopy(parallel_config)
        new_parallel_config.vocab_emb_dp = True

        self.norm = LayerNorm((config.hidden_size,))

    def construct(self, input_ids):
        """The forward compute of Embedding Layer."""
        word_embedding, word_table = self.word_embeddings(input_ids)
        embedding = self.norm(word_embedding)
        return embedding, word_table


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

class BloomModel(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.embedding = BloomEmbeddingLayer(config)
        self.embedding.pipeline_stage = 0


        self.get_attention_mask = AttentionMask(seq_length=config.seq_length,
                                                parallel_config=config.parallel_config.dp_mp_config)

        if not hasattr(config.parallel_config, "moe_config"):
            config.parallel_config.moe_config = default_moe_config
        moe_config = config.parallel_config.moe_config
        self.blocks = TransformerEncoder(hidden_size=config.hidden_size,
                                         batch_size=config.batch_size,
                                         ffn_hidden_size=config.hidden_size * config.expand_ratio,
                                         seq_length=config.seq_length,
                                         num_layers=config.num_layers,
                                         num_heads=config.num_heads,
                                         attention_dropout_rate=config.attention_probs_dropout_prob,
                                         hidden_dropout_rate=config.hidden_dropout_prob,
                                         hidden_act=config.hidden_act,
                                         lambda_func=set_parallel_configure_for_layer,
                                         param_init_type=config.param_init_type,
                                         layernorm_compute_type=config.layernorm_dtype,
                                         softmax_compute_type=config.softmax_dtype,
                                         parallel_config=config.parallel_config,
                                         moe_config=moe_config).blocks
        self.cast = P.Cast()
        self.tile = P.Tile().shard(((config.parallel_config.data_parallel,),))
        self.dtype = mstype.float16
        self.num_layers = config.num_layers
        self.input_position = Tensor(np.arange(config.seq_length), mstype.int32)

        self.ln_f = LayerNorm((config.hidden_size,)).to_float(config.layernorm_dtype)
        if config.parallel_config.pipeline_stage > 1:
            self.ln_f.set_comm_fusion(2)
        else:
            self.ln_f.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.ln_f.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.ln_f.pipeline_stage = config.parallel_config.pipeline_stage - 1


    def construct(self, input_ids, input_mask):
        """GPT model"""

        input_embedding, embedding_table = self.embedding(input_ids)

        hidden_states = self.cast(input_embedding, self.dtype)

        attention_mask = self.get_attention_mask(input_mask)

        for i in range(self.num_layers):
            hidden_states, _ = self.blocks[i](hidden_states, attention_mask)

        output_state = self.ln_f(hidden_states)

        return output_state, embedding_table


class BloomHead(nn.Cell):
    r"""Head for GPT to get the logits of each token in the vocab."""
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super().__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        mp = copied_parallel_config.model_parallel
        if vocab_size % mp != 0:
            logger.warning("The vocab size of GPTHead MatMul is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of GPTHead MatMul will be changed: mp = 1")
            copied_parallel_config.model_parallel = 1

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        if copied_parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (
                copied_parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()
        self.reshape = P.Reshape()

    def construct(self, state, embedding_table):
        state = self.reshape(state, (-1, self.hidden_size))
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embedding_table, self.dtype))
        return logits


class BloomLMHeadModel(BaseModel):
    r"""
        Provide gpt training loss or logits through network.
        Args:
            config (GPT2Config): The config of Gpt2Model.

        Returns:
            Tensor, the loss or logits of the network.
        """

    def __init__(self, config=None):
        config = config if config is not None else BloomConfig()
        super().__init__(config, auto_prefix=False)

        self.eos_token = self.config.eos_token
        parallel_config = self.config.parallel_config
        self.stridedslice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))

        self.backbone = BloomModel(self.config)
        self.head = BloomHead(hidden_size=config.hidden_size,
                            vocab_size=config.vocab_size,
                            parallel_config=self.config.parallel_config)
        if parallel_config.pipeline_stage > 1:
            self.head.pipeline_stage = parallel_config.pipeline_stage - 1
            self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of GPT Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of GPT Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1

        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.load_checkpoint(config)

    def construct(self, input_ids):
        r"""
            construct function for Language Modeling

            Args:
                input_ids (Tensor): the indices of input sequence tokens in the vocabulary.

            Returns:
                logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """

        batch_size, seq_length = input_ids.shape

        if self.phase == "train":
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
        else:
            tokens = input_ids

        input_mask = self.cast(self.not_equal(tokens, self.eos_token), mstype.float32)

        # [batch_size, seq_length, vocab_size]
        output_states, embedding_table = self.backbone(tokens, input_mask)
        logits = self.head(output_states, embedding_table)

        if self.phase != 'train':
            return logits

        labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss


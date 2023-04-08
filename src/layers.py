import math
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore import nn, ops
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
# from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindformers.modules.layers import LayerNorm, Linear, \
    _check_past_none_input_none, _check_input_dtype
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, _PipeLineConfig, OpParallelConfig, \
    _Config, _check_config, MoEParallelConfig
from mindformers.modules.transformer.moe import default_moe_config, MoE, _check_moe_config

from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.utils import is_version_ge
from mindformers.modules.transformer import FeedForward
from mindformers.modules.transformer.transformer import default_transformer_config, _get_lambda_func


class BloomAttention(Cell):
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config):
        super().__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dp = parallel_config.data_parallel
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if batch_size:
            Validator.check_positive_int(batch_size)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the num_heads is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(num_heads, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_channels=hidden_size,
                                     out_channels=hidden_size,
                                     transpose_b=False,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type)
            self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                                  strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                   (parallel_config.model_parallel, 1)))
            self.projection.bias.parallel_optimizer = False
            self.transpose = P.Transpose()
            self.merger_head_transpose = P.Transpose()
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            self.batch_matmul = P.BatchMatMul()
            self.real_div = P.RealDiv()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.add = P.Add()
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.size_per_head))
            self.beta = Tensor(1.0)
            self.use_past = use_past
            self.dropout = nn.Dropout(1 - hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(1 - attention_dropout_rate)
            self.softmax = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
            self.expand_dims = P.ExpandDims()

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the num_heads is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(num_heads, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_channels=hidden_size,
                                     out_channels=hidden_size,
                                     transpose_b=False,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type)
            self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                                  strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                   (parallel_config.model_parallel, 1)))
            self.projection.bias.parallel_optimizer = False
            self.transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))
            self.merger_head_transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            self.batch_matmul = P.BatchMatMul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.real_div = P.RealDiv().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
            self.sub = P.Sub().shard(
                ((1,), (parallel_config.data_parallel, 1, 1, 1)))
            self.mul = P.Mul().shard(
                ((parallel_config.data_parallel, 1, 1, 1), (1,)))
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.size_per_head))
            self.beta = Tensor(1.0)
            self.use_past = use_past
            self.dropout = nn.Dropout(1 - hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(1 - attention_dropout_rate)
            self.dropout.dropout.shard(((parallel_config.data_parallel, 1),))
            self.prob_dropout.dropout.shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax = nn.Softmax().to_float(softmax_compute_type)
            self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
            self.expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))

            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, query_tensor, key_tensor, value_tensor, alibi_tensor, attention_mask,
                  key_past=None, value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = query_tensor.shape
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        ori_dtype = query_tensor.dtype
        query_tensor = query_tensor.astype(self.dtype)
        key_tensor = key_tensor.astype(self.dtype)
        value_tensor = value_tensor.astype(self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            query.reshape((batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = self.transpose(
            key.reshape((batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                        self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            value.reshape((batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and attention_mask.ndim == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = (self.less(self.range, batch_valid_length.view(-1, 1, 1))).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = self.reducesum((self.not_equal(self.slice(key_past, (0, 0, 0, 0),
                                                                               (key_tensor.shape[0], 1, 1,
                                                                                self.src_seq_length),
                                                                               (1, 1, 1, 1)),
                                                                    0)).astype(mstype.float32), (1, 2, 3))
                valid_length = valid_length.reshape((-1, 1, 1))
                valid_length_vector = (self.equal(valid_length, self.range)).astype(self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = self.attention_mask.reshape((self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        attention = self._attn(query, key, value, alibi_tensor, attention_mask)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = output.reshape(ori_shape)
        output = output.astype(ori_dtype)
        return output, layer_present

    def _get_batch_size_from_query(self, query):
        r"""Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if query.ndim == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return query.shape[0] // self.src_seq_length
        return query.shape[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _check_inputs(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                      value_past=None, batch_valid_length=None):
        r"""Check inputs"""
        _check_input_dtype(query_tensor.dtype, "query_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(key_tensor.dtype, "key_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(value_tensor.dtype, "value_tensor", [mstype.float32, mstype.float16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(attention_mask.dtype, "attention_mask", [mstype.float32, mstype.float16],
                               self.cls_name)

        key_is_tensor = isinstance(key_past, Tensor)
        value_is_tensor = isinstance(value_past, Tensor)
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        key_is_default = key_past is None
        value_is_default = value_past is None
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "key_past", self.cls_name, None, key_is_tensor,
                                    key_is_default)
        _check_past_none_input_none(self.use_past, "value_past", self.cls_name, None, value_is_tensor,
                                    value_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)
        if self.use_past:
            _check_input_dtype(key_past.dtype, "key_past", [mstype.float16], self.cls_name)
            _check_input_dtype(value_past.dtype, "value_past", [mstype.float16], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor):
        """convert a nd tensor to a 2d tensor"""
        query_shape = query_tensor.shape
        query_tensor = query_tensor.reshape((-1, query_shape[-1]))
        key_shape = key_tensor.shape
        key_tensor = key_tensor.reshape((-1, key_shape[-1]))
        value_shape = value_tensor.shape
        value_tensor = value_tensor.reshape((-1, value_shape[-1]))

        return query_tensor, key_tensor, value_tensor

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = x.reshape(new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = attention_scores.shape
            # attention probs
            attention_probs = self.softmax_3d(
                attention_scores.reshape((shape[0], -1, shape[-1])))
            attention_probs = attention_probs.reshape(shape)
        return attention_probs

    def _attn(self, query, key, value, alibi_tensor, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = self.scale_factor.astype(query.dtype)
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)
        ori_dtype = score.dtype
        score = self.add(
            self.mul(score, self.inv_norm_factor.astype(ori_dtype)),
            self.mul(alibi_tensor, self.beta.astype(ori_dtype))
            )
        attention_scores = score.astype(self.softmax_dtype)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = self.reducesum((self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                                (query.shape[0], 1, 1,
                                                                                 self.seq_length),
                                                                                (1, 1, 1, 1)),
                                                                     0)).astype(mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(current_index.astype(mstype.int32), 1)
                index = index.reshape((-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = (self.tensor_le(self.range, index)).astype(mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub(
                Tensor((1.0,)).astype(attention_scores.dtype),
                attention_mask.astype(attention_scores.dtype))

            adder = self.mul(multiplu_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = attention_probs.astype(ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge



class BloomBlock(Cell):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super().__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                    .format(ffn_hidden_size, parallel_config.model_parallel))
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)

            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = BloomAttention(batch_size=batch_size,
                                                src_seq_length=seq_length,
                                                tgt_seq_length=seq_length,
                                                hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                attention_dropout_rate=attention_dropout_rate,
                                                softmax_compute_type=softmax_compute_type,
                                                param_init_type=param_init_type,
                                                use_past=use_past,
                                                parallel_config=attention_parallel_config)
            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          param_init_type=param_init_type,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                    .format(ffn_hidden_size, parallel_config.model_parallel))
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm1.shard(((parallel_config.data_parallel, 1),))
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2.shard(((parallel_config.data_parallel, 1),))

            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = BloomAttention(batch_size=batch_size,
                                                src_seq_length=seq_length,
                                                tgt_seq_length=seq_length,
                                                hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                attention_dropout_rate=attention_dropout_rate,
                                                softmax_compute_type=softmax_compute_type,
                                                param_init_type=param_init_type,
                                                use_past=use_past,
                                                parallel_config=attention_parallel_config)
            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          param_init_type=param_init_type,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, x, alibi_tensor, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = x.shape
        x = x.reshape((-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = input_x.astype(self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, init_reset.astype(self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, init_reset.astype(self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = ops.depend(input_x, key_reset)
            input_x = ops.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, alibi_tensor, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = output_x.astype(self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = ops.depend(key_update, key_reset)
            value_update = ops.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = ops.depend(mlp_logit, value_update)
        mlp_logit = ops.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = output_x.reshape(x_shape)
            mlp_logit = mlp_logit.reshape(x_shape)
            x = x.reshape(x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = output.reshape((-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = output.reshape(x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = output.reshape(x_shape)

        if self.use_moe:
            return output, layer_present, aux_loss
        return output, layer_present

    def _check_input(self, x, input_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        if input_mask is not None:
            _check_input_dtype(input_mask.dtype, "input_mask", [mstype.float32, mstype.float16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_input_dtype(init_reset.dtype, "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True



class BloomBlocks(Cell):
    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0,
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):
        super().__init__()
        _check_config(parallel_config)
        _check_moe_config(moe_config, parallel_config)
        self.use_moe = (moe_config.expert_num > 1)
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = BloomBlock(hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    ffn_hidden_size=ffn_hidden_size,
                                    seq_length=seq_length,
                                    attention_dropout_rate=attention_dropout_rate,
                                    hidden_dropout_rate=hidden_dropout_rate,
                                    layernorm_compute_type=layernorm_compute_type,
                                    softmax_compute_type=softmax_compute_type,
                                    num_heads=num_heads,
                                    hidden_act=hidden_act,
                                    post_layernorm_residual=post_layernorm_residual,
                                    param_init_type=param_init_type,
                                    use_past=use_past,
                                    moe_config=moe_config,
                                    parallel_config=config_to_layer)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = BloomBlock(hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    ffn_hidden_size=ffn_hidden_size,
                                    seq_length=seq_length,
                                    attention_dropout_rate=attention_dropout_rate,
                                    hidden_dropout_rate=hidden_dropout_rate,
                                    layernorm_compute_type=layernorm_compute_type,
                                    softmax_compute_type=softmax_compute_type,
                                    num_heads=num_heads,
                                    hidden_act=hidden_act,
                                    post_layernorm_residual=post_layernorm_residual,
                                    param_init_type=param_init_type,
                                    use_past=use_past,
                                    moe_config=moe_config,
                                    parallel_config=config_to_layer)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, hidden_states, alibi_tensor, attention_mask, init_reset=True, batch_valid_length=None):
        """forward process"""
        present_layer = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                hidden_states, present, aux_loss = self.blocks[i](hidden_states,
                                                                  attention_mask,
                                                                  init_reset,
                                                                  batch_valid_length)
                present_layer = present_layer + (present,)
                accum_loss = self.add(accum_loss, aux_loss)
            return hidden_states, present_layer, accum_loss

        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    alibi_tensor,
                                                    attention_mask,
                                                    init_reset,
                                                    batch_valid_length)
            present_layer = present_layer + (present,)

        return hidden_states, present_layer

from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.common.seed import _get_graph_seed
import mindspore.common.dtype as mstype

class Embedding(nn.Cell):
    """
        The embedding lookup table from the 0-th dim of the parameter table. When the parallel_config.vocab_emb_dp is
        True and in the `AUTO_PARALLEL` mode, the embedding lookup will be trained by the data parallel way, as the
        parameters will be repeated on each device. If false, the embedding table will be sharded into n parts at
        the 0-th dimension of the embedding table, where the n is the model parallel way determined by
        `parallel_config.model_parallel` (EmbeddingOpParallelConfig).

        Note:
            When `AUTO_PARALLEL` or `SEMI_AUTO_PARALLEL` mode is enabled, this layer support only 2-d dimension inputs,
            as the shard is designed for 2d inputs.

        Args:
            vocab_size (int): Size of the dictionary of embeddings.
            embedding_size (int): The size of each embedding vector.
            parallel_config (EmbeddingOpParallelConfig): The parallel config of network. Default
                `default_embedding_parallel_config`, an instance of `EmbeddingOpParallelConfig` with default args.
            param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.

        Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

        Outputs:
            Tuple, a tuple contains (`output`, `embedding_table`)

            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
            - **embedding_table** (Tensor) - The embedding table with shape (vocab_size, embedding_size).

        Raises:
            ValueError: If the parallel_config.vocab_emb_dp is True, the vocab size is not a multiple of
                parallel_config.model_parallel
            ValueError: `vocab_size` is not a positive value.
            ValueError: `embedding_size` is not a positive value.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import VocabEmbedding
            >>> from mindspore import Tensor
            >>> from mindspore import dtype as mstype
            >>> model = VocabEmbedding(vocab_size=30, embedding_size=30)
            >>> tensor = Tensor(np.ones((20, 15)), mstype.int32)
            >>> output, table = model(tensor)
            >>> print(output.shape)
            (20, 15, 30)
            >>> print(table.shape)
            (30, 30)
    """

    def __init__(self, vocab_size, embedding_size, param_init='normal'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_table = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                         name='embedding_table', parallel_optimizer=False)
        self.gather = ops.Gather()

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table.value()

    def shard(self, strategy):
        self.gather.shard(strategy)
        return self

class Linear(nn.Cell):
    r"""
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    3-D tensor.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    where :math:`X` is the input tensors, :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'. Default: None.
        compute_dtype (dtype.Number): The computation type. Default: mstype.float16
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 transpose_b=True,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(weight_init, Tensor) and (weight_init.ndim != 2 or weight_init.shape[0] != out_channels or
                                                weight_init.shape[1] != in_channels):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = ops.MatMul(transpose_b=transpose_b)
        self.bias_add = ops.Add()

        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias.parallel_optimizer = False
        self.dtype = compute_dtype

    def construct(self, x):
        """Forward process, x should be a tensor"""
        out_shape = x.shape[:-1] + (self.out_channels,)
        x = x.reshape((-1, self.in_channels))

        ori_dtype = x.dtype
        out = self.matmul(x.astype(self.dtype), self.weight.astype(self.dtype))
        out = out.astype(ori_dtype)

        if self.has_bias:
            out = self.bias_add(out, self.bias)

        out = out.reshape(out_shape)
        return out

    def shard(self, strategy_matmul, strategy_bias=None):
        r"""
        Set the shard for the linear. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy_matmul (tuple): The strategy for the matmul. Should be the same shape as the inputs.
            strategy_bias (tuple): The strategy for the bias_add. Should be the same shape as the inputs.
        """
        self.matmul.shard(strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        return self


class Dropout(nn.Cell):
    r"""
    Dropout layer for the input.

    Dropout is a regularization method. The operator randomly sets some neurons output to 0
    according to the probability of discarding the probability of discarding.
    During the reasoning, this layer returns the same Tensor as the `x`.

    This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
    over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
    preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Note:
        - Each channel will be zeroed out independently on every construct call.
        - Parameter `keep_prob` will be removed in a future version, please use parameter `p` instead.
          Parameter `p` means the probability of the element of the input tensor to be zeroed.
        - Parameter `dtype` will be removed in a future version. It is not recommended to define this parameter.

    Args:
        p (Union[float, int, None]): The dropout rate, greater than or equal to 0 and less than 1.
            E.g. rate=0.9, dropping out 90% of input neurons. Default: None.

    Inputs:
        - **x** (Tensor) - The input of Dropout with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, output tensor with the same shape as the `x`.

    Raises:
        TypeError: If `keep_prob` is not a float.
        TypeError: If the dtype of `p` is not float or int.
        TypeError: If dtype of `x` is not neither float16 nor float32.
        ValueError: If `keep_prob` is not in range (0, 1].
        ValueError: If `p` is not in range [0, 1).
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> net = nn.Dropout(p=0.2)
        >>> net.set_train()
        >>> output = net(x)
        >>> print(output.shape)
        (2, 2, 3)
    """

    def __init__(self, p=0.5):
        """Initialize Dropout."""
        super(Dropout, self).__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1), "
                                f"but got {p}.")
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.dropout = ops.Dropout(1.0 - p, seed0, seed1)
        self.p = p

    def construct(self, x):
        if not self.training or self.p == 0:
            return x

        out, _ = self.dropout(x)
        return out

    def extend_repr(self):
        return f'p={self.p}'

    def shard(self, strategy):
        self.dropout.shard(strategy)

        return self


# from gcn.inits import *
from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment 全局字典，用于为每个层分配唯一的 ID。
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """为层分配唯一的 ID。
    
    参数:
    layer_name (str): 层的名称。
    
    返回:
    int: 分配给层的唯一 ID。
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """对稀疏张量进行 Dropout 操作。
    
    参数:
    x (tf.SparseTensor): 输入的稀疏张量。
    keep_prob (float): 保留节点的概率。
    noise_shape (list): 噪声的形状。
    
    返回:
    tf.SparseTensor: 应用 Dropout 后的稀疏张量。
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """
    根据输入张量的类型（稀疏或密集）选择适当的矩阵乘法操作。
    参数:
    x (tf.Tensor or tf.SparseTensor): 第一个张量。
    y (tf.Tensor): 第二个张量。
    sparse (bool): 是否使用稀疏矩阵乘法。
    
    返回:
    tf.Tensor: 矩阵乘法的结果。
    """
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """基础层类，定义了所有层对象的基本 API。
    属性:
        name (str): 层的名称，定义了变量的作用域。
        logging (bool): 是否开启 Tensorflow 直方图日志。
    
    方法:
        _call(inputs): 定义层的计算图（即处理输入并返回输出）。
        __call__(inputs): _call() 的包装器。
        _log_vars(): 日志所有变量。
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        """定义层的计算图。
        参数:
        inputs (tf.Tensor): 输入张量。
        返回:
        tf.Tensor: 输出张量。
        """
        return inputs

    def __call__(self, inputs):
        """_call() 的包装器，添加命名空间和日志。
        参数:
        inputs (tf.Tensor): 输入张量。
        返回:
        tf.Tensor: 输出张量。
        """
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """连接层"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act  # 激活函数
        self.sparse_inputs = sparse_inputs  # 指示输入特征是否为稀疏张量（影响 Dropout 的实现方式）
        self.featureless = featureless      # featureless代表没有(/不用)节点特征信息，只用权重
        self.bias = bias    # 偏置

        # 稀疏 Dropout 的辅助变量
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # 权重初始化
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            # 偏置
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout：
        # 如果稀疏输入，使用 sparse_dropout 函数；
        # 对于密集输入，则使用 tf.nn.dropout 函数。
        # 这是因为稀疏张量的 Dropout 实现需要考虑非零元素的位置，而密集张量的 Dropout 可以直接在所有元素上操作
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """图卷积层"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        # 激活函数（relu）
        self.act = act
        
        # self.support 一个列表，包含了多个经过预处理的邻接矩阵（可能是原始邻接矩阵的不同幂次或经过归一化处理的版本
        # 例如，在Chebyshev图卷积网络中，k 可以代表Chebyshev多项式的阶数，每个阶数对应一个不同的邻接矩阵
        self.support = placeholders['support']
        
        self.sparse_inputs = sparse_inputs      # 指示输入特征是否为稀疏张量（影响 Dropout 的实现方式）
        self.featureless = featureless      # featureless代表没有(/不用)节点特征信息，只用权重
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # 权重初始化，每个W_i的维度：input_dim* output_dim
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))  
            # 偏置 
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        """
        定义图卷积层的计算图。
        参数:
        inputs (tf.Tensor): 输入张量。
        返回: σ(\sum_i (A_iXW_i) + b)
        tf.Tensor: 输出张量。
        
        A的维度: n*n
        X的维度: n*d
        W的维度: d*m (d为input_dim, m为output_dim)
        """
        x = inputs

        # dropout：
        # 如果稀疏输入，使用 sparse_dropout 函数；
        # 对于密集输入，则使用 tf.nn.dropout 函数。
        # 这是因为稀疏张量的 Dropout 实现需要考虑非零元素的位置，而密集张量的 Dropout 可以直接在所有元素上操作
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # 卷积过程
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)        # XW_i （有节点属性的情况）
            else:
                pre_sup = self.vars['weights_' + str(i)]        # W_i   （没有节点属性）
            support = dot(self.support[i], pre_sup, sparse=True)       # A_iXW_i
            supports.append(support)
        output = tf.add_n(supports)     # \sum_i (A_iXW_i)

        # bias
        if self.bias:
            output += self.vars['bias']     # \sum_i (A_iXW_i) + b

        return self.act(output)    # σ(\sum_i (A_iXW_i) + b)

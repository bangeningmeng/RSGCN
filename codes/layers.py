from codes.inits import *
import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()
tf.compat.v1.disable_eager_execution()
from crf import crf_layer
import time
import numpy as np

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:

        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
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
        return inputs

    def __call__(self, inputs):
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


class Encoder(Layer):
    """Encoder layer."""

    def __init__(self, size1, size2, latent_factor_num, placeholders, act=tf.nn.relu, featureless=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.act = act
        self.featureless = featureless
        self.size1 = size1
        self.size2 = size2
        self.latent_factor_num = latent_factor_num
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight1'] = glorot([size1[1] + size2[1], latent_factor_num])
            #             self.vars['weight1'] = glorot([size1[1]+size2[1],latent_factor_num])
            self.vars['weight2'] = glorot([size1[0] + size2[0], latent_factor_num])

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # convolution
        adj = inputs[0]

        print(adj.shape)
        feature = inputs[1]
        print(feature.shape)
        con = dot(adj, feature)

        # transform
        T = dot(con, self.vars['weight1'])
        hidden = T
        hidden = tf.nn.relu(tf.add(T, self.vars['weight2']))


        hidden_crf = hidden
        hidden_new = hidden
        for cv in range(0, 1):
            hidden_crf = crf_layer(hidden_crf, hidden_new)
            hidden_new = hidden_crf
        print("qqq")
        print(hidden_new.shape)

        return self.act(hidden_new)
        # return self.act(hidden)


class Decoder(Layer):
    """Decoder layer."""

    def __init__(self, size1, size2, latent_factor_num, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.size1 = size1
        self.size2 = size2
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight3'] = glorot([latent_factor_num, latent_factor_num])

    def _call(self, hidden):


        num_u = self.size1[0]
        U = hidden[0:num_u, :]
        V = hidden[num_u:, :]
        # U = ra_similarity(U)
        # V = ra_similarity(V)

        # U=tf(getGipKernel(hidden[0:,self.size1[0]], 0, 2 ** (-5), True).double())
        # V=tf(getGipKernel(hidden[self.size1[0]:], 0, 2 ** (-5), True).double())

        # getGipKernel(H1[:self.drug_size].clone(), 0, self.h1_gamma, True).double()

        # U = getGipKernel(hidden[0 : self.size1[0], : ], 0, 2 ** (-5), True).double()
        # V = getGipKernel(hidden[self.size1[0] : , : ], 0, 2 ** (-5), True).double()

        # print('*' * 30)
        # print(U.shape)
        # print(V.shape)
        # print(type(U),type(V))
        # time.sleep(10000)
        # print('*' * 30)

        M1 = dot(dot(U, self.vars['weight3']), tf.transpose(V))
        M1 = tf.reshape(M1, [-1, 1])
        return M1

def ra_similarity(x):
    ss=sum(x)
    ss[ss==0]=1
    result_1 = x /ss  # !!!!!sum默认是按照列求和
    y = np.array(result_1)  # 第一次传播--质量分配
    # print("第一次传播：\n",y)

    '''第2.1传播：均分'''
    result_2 = []
    y = np.squeeze(y)  # 去掉第一个维度   不知道为啥在k_yangzheng 调用后维度2变成3了
    for i in range(y.shape[0]):
        if x[i, :].sum() == 0:  # 训练测试集万一该列为空，避免相除报错
            continue
        result_2.append(y[i, :] / np.count_nonzero(y[i, :]))
    z = np.array(result_2)
    # print("第2.1次：\n", z)

    '''第2.2'''
    z_result = np.zeros((z.shape[1], z.shape[1]))
    for i in range(z.shape[1]):  # !!!!
        ii = np.where(z[:, i] > 0)
        # list(ii)[0]=每列不为0的索引  sum(axis=0)=数组按行相加
        z_result[i] = z[list(ii)[0], :].sum(axis=0)
    return z_result

def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = tf.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = tf.where(tf.math.is_inf(D_5), np.full_like(D_5, 0), D_5)
    L_D_11 = tf.matmul(D_5, L_D_1)
    L_D_11 = tf.matmul(L_D_11, D_5)
    return L_D_11


def normalized_embedding(embeddings):
    # [row, col] = embeddings.size()
    [row, col] = embeddings.shape

    # print('*' * 30)
    # print(type(embeddings))
    # print('*' * 30)

    ne = tf.Variable(tf.zeros([row, col]))
    # print('*' * 30)
    # print(type(ne))
    # print('*' * 30)


    for i in range(row):

        # print('*' * 30)
        # print((embeddings[i, :] - tf.reduce_min(embeddings[i, :])) / (tf.reduce_max(embeddings[i, :]) - tf.reduce_min(embeddings[i, :])))
        # # time.sleep(10000)
        # print('*' * 30)

        ne[i, : ] = (embeddings[i, :] - tf.reduce_min(embeddings[i, :])) / \
                    (tf.reduce_max(embeddings[i, :]) - tf.reduce_min(embeddings[i, :]))

    time.sleep(10000)
    return ne


def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = tf.matmul(y, y.T)
    krnl = krnl / tf.reduce_mean(tf.linalg.tensor_diag_part(krnl))
    krnl = tf.exp(-kernelToDistance(krnl) * gamma)
    return krnl

# def getGipKernel(y, trans, gamma, normalized=False):
#     if trans:
#         y = y.T
#     if normalized:
#         y = normalized_embedding(y)
#     krnl = t.mm(y, y.T)
#     krnl = krnl / t.mean(t.diag(krnl))
#     krnl = t.exp(-kernelToDistance(krnl) * gamma)
#     return krnl


def kernelToDistance(k):
    # di = tf.diag(k).T
    di = tf.transpose(tf.linalg.tensor_diag_part(k))
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


# def cosine_kernel(tensor_1, tensor_2):
#     return tf.DoubleTensor([tf.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
#                            range(tensor_1.shape[0])])

# def normalized_kernel(K):
#     K = abs(K)
#     k = K.flatten().sort()[0]
#     min_v = k[tf.nonzero(k, as_tuple=False)[0]]
#     K[tf.where(K == 0)] = min_v
#     D = tf.diag(K)
#     D = D.sqrt()
#     S = K / (D * D.T)
#     return S
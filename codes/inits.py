import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp


def uniform(shape, scale=0.5, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def get_weightmatrix(shape1, shape2, name=None):
    #     initial = tf.zeros(shape1, dtype=tf.float32)
    #     m1 = tf.Variable(initial, name=name)
    m1 = glorot(shape1)
    #     initial = tf.zeros(shape2, dtype=tf.float32)
    #     m2 = tf.Variable(initial, name=name)
    m2 = glorot(shape2)
    z1 = tf.constant(0, shape=shape1, dtype=tf.float32)
    z2 = tf.constant(0, shape=shape2, dtype=tf.float32)
    M = tf.concat([tf.concat([z2, m2], 1), tf.concat([m1, z1], 1)], 0)
    return M

# def laplacian(kernel):
#     d1 = sum(kernel)
#     D_1 = tf.diag(d1)
#     L_D_1 = D_1 - kernel
#     D_5 = D_1.rsqrt()
#     D_5 = tf.where(tf.isinf(D_5), tf.full_like(D_5, 0), D_5)
#     L_D_11 = tf.mm(D_5, L_D_1)
#     L_D_11 = tf.mm(L_D_11, D_5)
#     return L_D_11
#
#
# def normalized_embedding(embeddings):
#     [row, col] = embeddings.size()
#     ne = tf.zeros([row, col])
#     for i in range(row):
#         ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
#     return ne
#
#
# def getGipKernel(y, trans, gamma, normalized=False):
#     if trans:
#         y = y.T
#     if normalized:
#         y = normalized_embedding(y)
#     krnl = tf.mm(y, y.T)
#     krnl = krnl / tf.mean(tf.diag(krnl))
#     krnl = tf.exp(-kernelToDistance(krnl) * gamma)
#     return krnl
#
#
# def kernelToDistance(k):
#     di = tf.diag(k).T
#     d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
#     return d
#
#
# def cosine_kernel(tensor_1, tensor_2):
#     return tf.DoubleTensor([tf.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
#                            range(tensor_1.shape[0])])
#
#
# def normalized_kernel(K):
#     K = abs(K)
#     k = K.flatten().sort()[0]
#     min_v = k[tf.nonzero(k, as_tuple=False)[0]]
#     K[tf.where(K == 0)] = min_v
#     D = tf.diag(K)
#     D = D.sqrt()
#     S = K / (D * D.T)
#     return S

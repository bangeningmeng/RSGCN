import tensorflow.compat.v1 as tf
from codes.train import *
from codes.prepare import prepareData
from codes.poltting import poltting


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'GAutoencoder', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 110, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.9, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 70e-6, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') #最大切比雪夫多项式次数
flags.DEFINE_integer('latent_factor_num', 70, 'The size of latent factor vector.')

# K = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # M = [0.5, 0.6, 0.7, 0.8, 0.9]
# M = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# for k in K:
#     for m in M:
#
#         # Preparing data for training
#         labels, CS, DS, CF, DF, AM = prepareData(FLAGS,k,m)
#         reorder = np.arange(labels.shape[0])
#         np.random.shuffle(reorder)
#
#         T =8 # Number of training rounds
#         cv_num = 9 # k-flod Cross-validation (CV)
#         for t in range(T):
#             test_labels = []
#             score = []
#             test_acc = []
#             order = div_list(reorder.tolist(), cv_num)
#             for i in range(cv_num):
#                 print("K:{} M:{}".format(str(k),str(m)))
#                 print("T:", '%01d' % (t))
#                 print("cross_validation:", '%01d' % (i))
#                 test_arr = order[i]
#                 arr = list(set(reorder).difference(set(test_arr)))
#                 np.random.shuffle(arr)
#                 train_arr = arr
#                 test_labels1, score1, test_acc1 = train(FLAGS, train_arr, test_arr, labels, CS, DS, CF, DF, )
#                 test_labels.append(test_labels1)
#                 score.append(score1)
#                 test_acc.append(test_acc1)
#             poltting(test_labels, score, t, cv_num, test_acc, k, m)
#Preparing data for training
labels, CS, DS, CF, DF, AM = prepareData(FLAGS)
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

T =8 # Number of training rounds
cv_num = 10 # k-flod Cross-validation (CV)
for t in range(T):
    test_labels = []
    score = []
    test_acc = []
    order = div_list(reorder.tolist(), cv_num)
    for i in range(cv_num):
        # print("K:{} M:{}".format(str(k),str(m)))
        print("T:", '%01d' % (t))
        print("cross_validation:", '%01d' % (i))
        test_arr = order[i]
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr
        test_labels1, score1, test_acc1 = train(FLAGS, train_arr, test_arr, labels, CS, DS, CF, DF)
        test_labels.append(test_labels1)
        score.append(score1)
        test_acc.append(test_acc1)
    poltting(test_labels, score, t, cv_num, test_acc)

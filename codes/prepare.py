import numpy as np
import pandas as pd
from tqdm import trange
from codes.rwr import SimtoRWR
from codes.pca import reduction
from crf import crf_layer
from sklearn.metrics.pairwise import cosine_similarity


def Makeadj(AM):
    adj = []
    for i in trange(AM.shape[0]):
        for j in range(AM.shape[1]):
            adj_inner = []
            if AM[i][j] == 1:
                adj_inner.append(i + 1)
                adj_inner.append(j + 1)
                adj_inner.append(1)
                adj.append(adj_inner)
    return np.array(adj)


def heteg(SC, SD, AM):
    reSC = np.hstack((SC, AM))
    reSD = np.hstack((SD, AM.T))
    return reSC, reSD
# def heteg(SC, SD, CS, DS):
#     reSC = np.hstack((SC, CS))
#     reSD = np.hstack((SD, DS))
#     return reSC, reSD

# def prepareData(FLAGS,k,m):
def prepareData(FLAGS):
   

    AM = np.load('../dataset/药物-靶标相互作用数据的邻接矩阵/ic_admat_dgc.npy')

    a = np.load('../dataset/药物结构相似矩阵/ic_simmat_dc.npy')

    c = np.load('../dataset/蛋白序列相似矩阵/ic_simmat_dg.npy')

    b = np.load('../dataset/药物-靶标相互作用数据的邻接矩阵/ic_admat_dgc_rasim_d.npy')

    d = np.load('../dataset/药物-靶标相互作用数据的邻接矩阵/ic_admat_dgc_rasim_p.npy')
    
    adj = Makeadj(AM)
    
    # Using RWR to calculate CRS and DRS
    CRS, DRS = SimtoRWR(CS, DS, FLAGS)
    
    CRS = cosine_similarity(CRS)
    DRS = cosine_similarity(DRS)

    # Matrix Splicing
    reSC, reSD = heteg(CRS, DRS, AM)
    

    # Matrix noise reduction and dimensionality reduction
    CF, DF = reduction(reSC, reSD, FLAGS)
    return adj, CS, DS, CF, DF, AM


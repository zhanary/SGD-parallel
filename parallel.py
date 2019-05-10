# encoding=utf8

#######################
# Author: ZouHang
# StudentID: 1809853P-II20-0032
# Date: 2019-05-10
#######################

import numpy as np
import math

import multiprocessing
from multiprocessing import Process
from multiprocessing.sharedctypes import Value, RawArray
kernels=multiprocessing.cpu_count()

def SGDparallel(iter_num, testMat, P_base, Q_base, m, b1, b2, offset_m, offset_n):
    K = 5
    row, col = np.shape(testMat)
    u_value=m.value
    m = np.mean(testMat)
    P=np.mat(np.frombuffer(P_base).reshape((row,K)))
    Q=np.mat(np.frombuffer(Q_base).reshape((K,col)))
    alpha = 0.0004
    beta = 0.01

    sample_offset_m=np.linspace(0,row,kernels+1).astype('int')

    sample_offset_n=np.linspace(0,col,kernels+1).astype('int')

    for step in range(iter_num):
        for i in range(sample_offset_m[offset_m],sample_offset_m[offset_m+1]):
            for j in range(sample_offset_n[offset_n],sample_offset_n[offset_n+1]):
                if testMat[i,j] > 0:
                    tmp=0.0
                    for k in range(K):
                        tmp = tmp + P[i,k]*Q[k,j]
                    error = testMat[i,j] - tmp - m - b1[i] - b2[j]
                    b1[i] = b1[i] + alpha * (error - beta * b1[i])
                    b2[j] = b2[j] + alpha * (error - beta * b2[j])
                    for k in range(K):
                        P[i,k] = P[i,k] + alpha * (error * Q[k,j] - beta * P[i,k])
                        Q[k,j] = Q[k,j] + alpha * (error * P[i,k] - beta * Q[k,j])
    for i in range(sample_offset_m[offset_m],sample_offset_m[offset_m+1]):
        for k in range(K):
            P_base[i*K+k] = P[i,k]
    for k in range(K):
        for j in range(sample_offset_n[offset_n],sample_offset_n[offset_n+1]):
            Q_base[k*col+j] = Q[k,j]
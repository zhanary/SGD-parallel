# encoding=utf8

#######################
# Author: ZouHang
# StudentID: 1809853P-II20-0032
# Date: 2019-05-10
#######################

import numpy as np
import math

def SGDnormal(iter_num, testMat1, baseMat1):
    K = 5
    row, col = np.shape(baseMat1)
    testMat=np.mat(testMat1)
    baseMat=np.mat(baseMat1)
    P = np.mat(np.zeros((row, K), np.float32))
    Q = np.mat(np.zeros((K, col), np.float32))
    m = np.mean(testMat)
    b1 = np.mean(testMat, axis = 1)
    b2 = np.mean(testMat, axis = 0).T

    alpha = 0.0004
    beta = 0.01
    iter_num = 100
    #print(testMat[235])
    for step in range(iter_num):
        #print(step)
        for i in range(row):
            for j in range(col):
                if testMat[i,j] > 0:
                    tmp=0.0
                    for k in range(K):
                        tmp = tmp + P[i,k]*Q[k,j]
                    error = testMat[i,j] - tmp - m - b1[i,0] - b2[j,0]
                    b1[i,0] = b1[i,0] + alpha * (error - beta * b1[i,0])
                    b2[j,0] = b2[j,0] + alpha * (error - beta * b2[j,0])
                    for k in range(K):
                        P[i,k] = P[i,k] + alpha * (error * Q[k,j] - beta * P[i,k])
                        Q[k,j] = Q[k,j] + alpha * (error * P[i,k] - beta * Q[k,j])

    return P, Q, b1, b2
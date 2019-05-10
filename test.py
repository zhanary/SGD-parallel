# encoding=utf8

#######################
# Author: ZouHang
# StudentID: 1809853P-II20-0032
# Date: 2019-05-10
#######################

import numpy as np
import math
import time
import multiprocessing
from multiprocessing import Process
from multiprocessing.sharedctypes import Value, RawArray
kernels=multiprocessing.cpu_count()
print("kernels: ", kernels)

from dataTool import load_data
from SGD import SGDnormal
from parallel import SGDparallel



if __name__ == "__main__":
    ###########define###########
    basedata_name = "u1.base"
    testdata_name = "u1.test"
    iter_num = 50
    ############################
    print("Iter_num is = ", iter_num)
    baseData, row, col = load_data(basedata_name, 0, 0)
    testData, row, col = load_data(testdata_name, row, col)
    print(row,col)
    baseMat = np.mat(baseData)
    testMat = np.mat(testData)
    m = np.mean(testMat)

    single_begin = time.time()

    P, Q, b1, b2 = SGDnormal(iter_num, testData, baseData)
    result = np.zeros((row, col), np.float32)
    PQ = P * Q

    for i in range(row):
        for j in range(col):
            result[i, j] = m + b1[i, 0] + b2[j, 0] + PQ[i, j]
    root = 0.0
    count = 0
    for i in range(row):
        for j in range(col):
            if baseMat[i, j] > 0:
                root = root + np.square(baseMat[i, j] - result[i, j])
                count = count + 1
    RMSE = math.sqrt(root / count)
    print("RMSE - SGD = ", RMSE)

    timeConsume1 = (time.time() - single_begin)
    print("time - SGD consumed = ", timeConsume1)

    testData, row, col = load_data(testdata_name, row, col)
    testMat = np.mat(testData)

    multi_begin = time.time()
    K = 5
    P_base = RawArray('d', row * K)
    Q_base = RawArray('d', K * col)

    b1 = np.mean(testMat, axis=1)
    b2 = np.mean(testMat, axis=0).T

    b1_base = RawArray('f', b1.flat)
    b2_base = RawArray('f', b2.flat)

    u_base = Value('d', m, lock=False)

    j = 0
    for k in range(kernels):
        record = []
        for i in range(kernels):
            process = Process(target=SGDparallel,
                              args=(iter_num, testData, P_base, Q_base, u_base, b1_base, b2_base, (i + k) % kernels, j % kernels))
            j += 1
            process.start()
            record.append(process)
        for process in record:
            process.join()


    P = np.mat(np.frombuffer(P_base).reshape((row, K)))
    Q = np.mat(np.frombuffer(Q_base).reshape((K, col)))
    b1 = np.mat(np.frombuffer(b1_base, dtype=np.float32).reshape((1, row)))
    b2 = np.mat(np.frombuffer(b2_base, dtype=np.float32).reshape((1, col)))
    result = np.zeros((row, col), np.float32)
    PQ = P * Q
    for i in range(row):
        for j in range(col):
            result[i, j] = m + b1[0, i] + b2[0, j] + PQ[i, j]
    root = 0.0
    count = 0
    for i in range(row):
        for j in range(col):
            if baseMat[i, j] > 0:
                root = root + np.square(baseMat[i, j] - result[i, j])
                count = count + 1
    RMSE = math.sqrt(root / count)
    print("RMSE - parallel = ", RMSE)

    timeConsume2 = (time.time() - multi_begin)
    print("time - parallel = ", timeConsume2)
    print("R = T0 / Tp = %.2f / %.2f = %.2f" % (timeConsume1, timeConsume2, timeConsume1 / timeConsume2))




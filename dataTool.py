# encoding=utf8

#######################
# Author: ZouHang
# StudentID: 1809853P-II20-0032
# Date: 2019-05-10
#######################

import numpy as np
import math

def load_data(path, row, col):
    with open(path) as f:
        row_num = 0
        col_num = 0
        for line in f.readlines():
            user = int(line.strip().split('\t')[0])
            if user > row_num:
                row_num = user
            movie = int(line.strip().split('\t')[1])
            if movie > col_num:
                col_num = movie
        if row > row_num:
            row_num = row
        if col > col_num:
            col_num = col
        data = np.zeros((row_num, col_num), np.float32)
        f = open(path)
        for line in f.readlines():
            user, movie, rate, _ = (int(i) for i in line.strip().split('\t'))
            data[user-1,movie-1] = rate
    return data, row_num, col_num
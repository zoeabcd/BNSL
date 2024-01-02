import numpy as np
import random

def Get_Data_Count(i, qi, K, D):
    # return N, where N[j, k] = # of instance that Xi in k, K is in j (j in range(qi))
    N = np.zeros((qi, 2))
    for data in D:
        num = 0
        idx = 0
        for j in data[K]:
            num += j * (2 ** idx)
            idx += 1
        datai = (int)(data[i])
        num = (int)(num)
        N[num, datai] += 1
    return N

def subscore(i, K, D):
    # compute s_i(K) for graph G and data D
    res = 0
    qi = 2 ** len(K)
    N = Get_Data_Count(i, qi, K, D)
    # N[i, j, k] = # of instance that Xi in k, K is in j (0-3 in this example) and for simplicity, omit i here.
    for j in range(qi):
        for k in range(1, int(N[j, 0] + N[j, 1] + 1) + 1):
            res += np.log(k)
        for k in range(1, int(N[j, 0]) + 1):
            res -= np.log(k)
        for k in range(1, int(N[j, 1]) + 1):
            res -= np.log(k)
    return res

def score(G, D, n):
    # G[i, j] = 1 means arc (i, j) exist
    res = 0
    for i in range(n):
        K = np.argwhere(G[:, i] == 1).transpose()[0]
        res += subscore(i, K, D)
    return res

def Read_Data(txtname, n):
    # the first line of the file is the colume idx of features
    f = open(txtname)
    line = f.readlines()
    num_line = len(line)

    D = np.zeros((num_line, n))
    idx = 0
    for li in line:
        list = li.strip('\n').split(' ')
        D[idx,:] = list[0:n]
        idx += 1
    
    return D[1:, :]
    



def Generate_Data(n, M = 200):
    # M = # of samples
    D = np.zeros((M, n))
    if n == 3:
        for idx in range(M):
            if random.uniform(0, 1) > 0.2:
                D[idx, 0] = 1
                if random.uniform(0, 1) > 0.8:
                    D[idx, 1] = 1
                if random.uniform(0, 1) > 0.2:
                    D[idx, 2] = 1
    elif n == 2:
        for idx in range(M):
            if random.uniform(0, 1) > 0.5:
                D[idx, 0] = 1
                if random.uniform(0, 1) > 0.5:
                        D[idx, 1] = 1
    elif n == 4:
        for idx in range(M):
            if random.uniform(0, 1) > 0.5:
                D[idx, 0] = 1
                if random.uniform(0, 1) > 0.9:
                    D[idx, 1] = 1
                    if random.uniform(0, 1) > 0.9:
                        D[idx, 2] = 1
                        if random.uniform(0, 1) > 0.9:
                            D[idx, 3] = 1
    else: 
        for idx in range(M):
            D[idx, 0] = 1
            for k in range(1, n):
                if random.uniform(0, 1) > random.uniform(0, 1) and D[idx, k-1] == 1 :
                    D[idx, k] = 1
    return D

def powerset(J):
    if len(J) == 0:
        return [[]]
    sub_J = J[1:]
    sub_ps = powerset(sub_J)
    res = sub_ps.copy()
    for sub in sub_ps:
        res.append([J[0]] + sub)
    return res

def omega(i, J, D):
    p_set = powerset(J)
    L = len(J)
    res = 0
    for K in p_set:
        tmp = L - len(K)
        if tmp % 2 == 0:
            res += subscore(i, K, D)
        else:
            res -= subscore(i, K, D)
    return res
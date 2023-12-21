from sympy import *
from data_score import non_empty_powerset, omega
import numpy as np

def H_subscore(i, n, m, D, d, y, r):
    res = 0
    tmp = list(range(n))
    del tmp[i]
    possible_parent_sets = non_empty_powerset(tmp)
    for J in possible_parent_sets:
        if len(J) > m: continue 
        tmp = 1
        for j in J:
            tmp *= d[j, i]
        res += omega(i, J, D) * tmp
    res = simplify(expand(res))
    return res

def H_score(n, m, D, d, y, r):
    res = 0
    for i in range(n):
        res += H_subscore(i, n, m, D, d, y, r)
    res = simplify(expand(res))
    return res

def H_max(n, m, delta_max, d, y, r):
    res = 0
    for i in range(n):
        di = np.sum(d[:,i]) - d[i, i]
        res += delta_max * \
            ((m - di - (y[i, 0] + 2*y[i, 1])) ** 2)
    res = simplify(expand(res))
    return res

def H_cons(n, delta_cons, d, y, r):
    res = 0
    for i in range(n-1):
        for j in range(i+1, n):
            idx = int(i*(n - 1) - i*(i+1)/2 + j - 1)
            res += delta_cons * \
            (d[i,j] * r[idx] + d[j,i] * (1 - r[idx]))
    res = simplify(expand(res))
    return res

def H_trans(n, delta_trans, d, y, r):
    res = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            idx_ij = int(i*(n-1) - i*(i+1)/2 + j - 1)
            for k in range(j+1, n):
                idx_ik = int(i*(n-1) - i*(i+1)/2 + k - 1)
                idx_jk = int(j*(n-1) - j*(j+1)/2 + k - 1)
                # print(i, j, k, n)
                # print(idx_ij, idx_ik, idx_jk)
                res += delta_trans * \
                    (r[idx_ik] + r[idx_ij] * r[idx_jk] \
                      - r[idx_ij]*r[idx_ik] - r[idx_jk]*r[idx_ik])
    res = simplify(expand(res))
    return res

def num_to_symbol(num, n, d, y, r):
    if num < n*(n-1) :
        i = int(num / (n-1))
        j = num % (n-1)
        j += 1 if j >= i else 0
        return d[i, j]
    elif num >= n*(n-1) and num < 3*n*(n-1)/2:
        num -= n*(n-1)
        return r[num]
    else:
        num -= int(3*n*(n-1)/2)
        i = int(num / 2)
        j = num % 2
        return y[i, j]


def hamitonian_para(n, m, D, delta_max, delta_cons, delta_trans):
    d = MatrixSymbol("d", n, n)
    y = MatrixSymbol("y", n, 2)
    r = MatrixSymbol("r", int(n*(n-1)/2), 1) # only up-right 

    res =   H_score(n, m, D, d, y, r) + \
            H_max(n, m, delta_max, d, y, r) + \
            H_cons(n, delta_cons, d, y, r) + \
            H_trans(n, delta_trans, d, y, r)
    res = simplify(expand(res))

    # res = res.subs({d[:,:] ** 2 : d[:,:]})
    # res = res.subs({y[:,:] ** 2 : y[:,:]})

    # res = res.subs({d[:, :] : (d[:, :]+1)/2})
    # res = res.subs({y[:, :] : (y[:, :]+1)/2})
    # res = res.subs({r[:] : (r[:]+1)/2})

    for i in range(n):
        for j in range(n):
            res = res.subs({d[i, j] ** 2 : d[i, j]})
            res = res.subs({d[i, j] : (d[i, j]+1)/2})

    for i in range(n):
        for j in range(2):
            res = res.subs({y[i, j] ** 2 : y[i, j]})
            res = res.subs({y[i, j] : (y[i, j]+1)/2})

    for i in range(int(n*(n-1)/2)):
        res = res.subs({r[i] : (r[i]+1)/2})

    res = simplify(expand(res))

    N = int(3*n*(n-1)/2 + 2*n)
    C = 0
    h = np.zeros(N)
    J = np.zeros((N, N))

    tmp = res.subs({y : ZeroMatrix(n, 2)})
    tmp = tmp.subs({d : ZeroMatrix(n, n)})
    tmp = tmp.subs({r : ZeroMatrix(int(n*(n-1)/2), 1)})
    tmp = simplify(expand(tmp))
    C = tmp

    for i in range(N):
        symi = num_to_symbol(i, n, d, y, r)
        tmp = Poly(res, symi).all_coeffs()
        tmp = tmp[::-1]
        if(len(tmp) >= 2):
            tmp = tmp[1]
            tmp = tmp.subs({y : ZeroMatrix(n, 2)})
            tmp = tmp.subs({d : ZeroMatrix(n, n)})
            tmp = tmp.subs({r : ZeroMatrix(int(n*(n-1)/2), 1)})
            tmp = simplify(expand(tmp))
            h[i] = tmp
        
    for i in range(N):
        for j in range(i + 1, N):
            symi = num_to_symbol(i, n, d, y, r)
            symj = num_to_symbol(j, n, d, y, r)
            tmp = Poly(res, symi).all_coeffs()
            tmp = tmp[::-1]
            if(len(tmp) >= 2):
                tmp = tmp[1]
                tmp = Poly(tmp, symj).all_coeffs()
                tmp = tmp[::-1]
                if(len(tmp) >= 2):
                    tmp = tmp[1]
                    tmp = tmp.subs({y : ZeroMatrix(n, 2)})
                    tmp = tmp.subs({d : ZeroMatrix(n, n)})
                    tmp = tmp.subs({r : ZeroMatrix(int(n*(n-1)/2), 1)})
                    tmp = simplify(expand(tmp))
                    J[i,j] = tmp
    
    return C,h,J

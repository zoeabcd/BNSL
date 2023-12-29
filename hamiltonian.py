from sympy import *
from data_score import powerset, omega, subscore
import numpy as np
import heapq
from operator import itemgetter
from tqdm import tqdm

def dist(i, n, J, d):
    st = set(J)
    res = n - 1
    for j in range(n):
        if j == i: continue
        if j in st:
            res -= d[j, i]
        else:
            res -= (1 - d[j, i])
    return res

def H_subscore_onelocal(i, n, m, D, d, y, r):
    res = 0
    tmp = list(range(n))
    del tmp[i]
    possible_parent_sets = powerset(tmp)
    for J in possible_parent_sets:
        if len(J) > m: continue
        res += subscore(i, J, D) * ((n - 1) / 2 - dist(i, n, J, d))
    res = simplify(expand(res))
    return res

def H_score_onelocal(n, m, D, d, y, r):
    res = 0
    for i in range(n):
        res += H_subscore_onelocal(i, n, m, D, d, y, r)
    res = simplify(expand(res))
    return res

def H_subscore(i, n, m, D, d, y, r):
    res = 0
    tmp = list(range(n))
    del tmp[i]
    possible_parent_sets = powerset(tmp)
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

def calculate_Delta_ji(n, m, D):
    Delta_ji = np.zeros((n,n))
    if m==1:
        for i in range(n):
            for j in range(n):
                Delta_ji[i, j] = -omega(i, [j], D)
    elif m==2:
        for i in range(n):
            for j in range(n):
                Delta_ji[i, j] = -omega(i, [j], D) - sum(min(0, omega(i, [j, k], D)) for k in range(n) if k != i and k != j)
    else:
        for i in range(n):
            for j in range(n):
                sum_ = 0
                # Generate all subsets of size up to m-2 to ensure |J| < m-1
                for J in itertools.chain.from_iterable(itertools.combinations(range(n), r) for r in range(m-1)):
                    # Convert tuple to list and ensure i, j not in J
                    J = list(J)
                    if i not in J and j not in J:
                        # Calculate omega with J union {j}
                        sum_ += min(0, omega(i, J + [j], D))
                Delta_ji[i, j] = -sum_
    for i in range(n):
        for j in range(n):
            Delta_ji[i, j] = max(0, Delta_ji[i, j])
    return Delta_ji

def generate_delta(Delta_ji):
    # Initialize the delta arrays
    n = Delta_ji.shape[0]
    delta_max_i = np.zeros(n)
    delta_consist_ij = np.zeros((n, n))
    delta_trans_ijk = np.zeros((n, n, n))

    # Calculate delta_max for each i
    for i in range(n):
        delta_max_i[i] = max(Delta_ji[i, j] for j in range(n) if j != i)

    # Calculate delta_consist for each pair (i, j)
    for i in range(n):
        for j in range(n):
            if i != j:
                delta_consist_ij[i, j] = (n - 2) * max(Delta_ji[i, k] for k in range(n) if k != i and k != j)

    # Calculate delta_trans for each triplet (i, j, k)
    # Given delta_trans_ijk = delta_trans > max Delta_ij' for all i != j'
    # We interpret this as delta_trans_ijk being a constant value for all i, j, k
    # where i < j < k, which is greater than any Delta_ij' where i != j'
    # Here, we use a conservative approach by taking the max of all Delta_ij' where i != j'
    delta_trans = max(max(Delta_ji[i, j] for j in range(n) if i != j) for i in range(n))

    # Assign delta_trans to every element in the delta_trans_ijk array
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i < j < k:
                    delta_trans_ijk[i, j, k] = delta_trans

    return delta_max_i, delta_consist_ij, delta_trans_ijk
    
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


def hamiltonian_para(n, m, D, delta_max, delta_cons, delta_trans, show_BF, onelocal):
    # one local is true == use approx H_score
    d = MatrixSymbol("d", n, n)
    y = MatrixSymbol("y", n, 2)
    r = MatrixSymbol("r", int(n*(n-1)/2), 1) # only up-right 

    if onelocal:
        res =  H_score_onelocal(n, m, D, d, y, r)
    else:
        res =  H_score(n, m, D, d, y, r)

    res =   H_max(n, m, delta_max, d, y, r) + \
            H_cons(n, delta_cons, d, y, r) + \
            H_trans(n, delta_trans, d, y, r)
    res = simplify(expand(res))

    for i in range(n):
        for j in range(n):
            res = res.subs({d[i, j] ** 2 : d[i, j]})

    for i in range(n):
        for j in range(2):
            res = res.subs({y[i, j] ** 2 : y[i, j]})

    res = simplify(expand(res))
    

    N = int(3*n*(n-1)/2 + 2*n)

    if show_BF:
        print("Before spin transformation:", res)
        bf_results = {}
        for x in tqdm(range(1 << N)):
            origx = x
            res2 = res.copy()
            for i in range(N):
                res2 = res2.subs({num_to_symbol(i, n, d, y, r): x & 1})
                x >>= 1
            res2 = simplify(expand(res2))
            bf_results["{:07b}".format(origx)] = float(res2)
        print("Brute force results:", dict(heapq.nsmallest(5, bf_results.items(), key=itemgetter(1))))

    for i in range(n):
        for j in range(n):
            res = res.subs({d[i, j] : (d[i, j]+1)/2})

    for i in range(n):
        for j in range(2):
            res = res.subs({y[i, j] : (y[i, j]+1)/2})

    for i in range(int(n*(n-1)/2)):
        res = res.subs({r[i] : (r[i]+1)/2})
    
    res = simplify(expand(res))

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

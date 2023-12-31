from sympy import *
from data_score import powerset, omega, subscore
from analysis_toolkit import combinations
import numpy as np
import heapq
from operator import itemgetter
from tqdm import tqdm
import random

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

def H_subscore_onelocal(i, n, m, D, d, y, r):
    res = 0
    tmp = list(range(n))
    del tmp[i]
    possible_parent_sets = powerset(tmp)
    for J in possible_parent_sets:
        if len(J) > m: continue
        res += subscore(i, J, D) * ((n - dist(i, n, J, d)) / n) ** 2
    res = simplify(expand(res))
    return res

def H_score(n, m, D, d, y, r):
    res = 0
    for i in range(n):
        res += H_subscore(i, n, m, D, d, y, r)
    res = simplify(expand(res))
    return res

def H_score_onelocal(n, m, D, d, y, r):
    res = 0
    for i in range(n):
        res += H_subscore_onelocal(i, n, m, D, d, y, r)
    res = simplify(expand(res))
    return res

def H_max(n, m, delta_max_i, d, y, r):
    if y is None:
        return 0
    res = 0
    for i in range(n):
        di = np.sum(d[:,i]) - d[i, i]
        res += delta_max_i[i] * \
            ((m - di - (y[i, 0] + 2*y[i, 1])) ** 2)
    res = simplify(expand(res))
    return res

def H_cons(n, delta_cons_ij, d, y, r):
    res = 0
    for i in range(n-1):
        for j in range(i+1, n):
            idx = int(i*(n - 1) - i*(i+1)/2 + j - 1)
            res += delta_cons_ij[i, j] * \
            (d[i,j] * r[idx] + d[j,i] * (1 - r[idx]))
    res = simplify(expand(res))
    return res

def H_trans(n, delta_trans_ijk, d, y, r):
    res = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            idx_ij = int(i*(n-1) - i*(i+1)/2 + j - 1)
            for k in range(j+1, n):
                idx_ik = int(i*(n-1) - i*(i+1)/2 + k - 1)
                idx_jk = int(j*(n-1) - j*(j+1)/2 + k - 1)
                res += delta_trans_ijk[i, j, k] * \
                    (r[idx_ik] + r[idx_ij] * r[idx_jk] \
                      - r[idx_ij]*r[idx_ik] - r[idx_jk]*r[idx_ik])
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
                for r in range(m-1):
                    for J in combinations(range(n), r):
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

def generate_delta(Delta_ji, factor=1.5, min_value=10):
    # Initialize the delta arrays
    n = Delta_ji.shape[0]
    delta_max_i = np.zeros(n)
    delta_consist_ij = np.zeros((n, n))
    delta_trans_ijk = np.zeros((n, n, n))

    # Calculate delta_max for each i
    for i in range(n):
        delta_max_i[i] = max(Delta_ji[i, j] for j in range(n) if j != i)

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

    # Calculate delta_consist for each pair (i, j)
    for i in range(n):
        for j in range(n):
            if i < j:
                if n > 2:
                    delta_consist_ij[i, j] = (n - 2) * max(delta_trans_ijk[i, j, k] for k in range(n) if k != i and k != j)
                else:
                    delta_consist_ij[i, j] = delta_trans

    return np.maximum(factor * delta_max_i, min_value), \
            np.maximum(factor * delta_consist_ij, min_value),\
            np.maximum(factor * delta_trans_ijk, min_value)

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

def hamiltonian_para(n, m, D, delta_max, delta_cons, delta_trans, show_BF, onelocal, use_y = True):
    # one local is true == use approx H_score
    d = MatrixSymbol("d", n, n)
    y = None
    if use_y:
        y = MatrixSymbol("y", n, 2)
    r = MatrixSymbol("r", int(n*(n-1)/2), 1) # only up-right 

    if onelocal:
        res =  H_score_onelocal(n, m, D, d, y, r)
    else:
        res =  H_score(n, m, D, d, y, r)

    res +=   H_max(n, m, delta_max, d, y, r) + \
            H_cons(n, delta_cons, d, y, r) + \
            H_trans(n, delta_trans, d, y, r)
    res = simplify(expand(res))

    for i in range(n):
        for j in range(n):
            res = res.subs({d[i, j] ** 2 : d[i, j]})

    if use_y:
        for i in range(n):
            for j in range(2):
                res = res.subs({y[i, j] ** 2 : y[i, j]})

    res = simplify(expand(res))
    print(res)
    

    N = int(3*n*(n-1)/2 )
    if use_y:
        N += 2 * n

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

    if use_y:
        for i in range(n):
            for j in range(2):
                res = res.subs({y[i, j] : (y[i, j]+1)/2})

    for i in range(int(n*(n-1)/2)):
        res = res.subs({r[i] : (r[i]+1)/2})
    
    res = simplify(expand(res))

    C = 0
    h = np.zeros(N)
    J = np.zeros((N, N))

    tmp = res
    if use_y:
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
            if use_y:
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
                    if use_y:
                        tmp = tmp.subs({y : ZeroMatrix(n, 2)})
                    tmp = tmp.subs({d : ZeroMatrix(n, n)})
                    tmp = tmp.subs({r : ZeroMatrix(int(n*(n-1)/2), 1)})
                    tmp = simplify(expand(tmp))
                    J[i,j] = tmp
    
    return C,h,J

def stochastic_normalize(h, J, samples=100):
    results = []
    for _ in range(samples):
        x = random.randrange(1 << len(h))
        values = np.zeros((len(h), ))
        for i in range(len(h)):
            values[i] = -1 if x & 1 != 0 else 1
            x >>= 1
        results.append(np.abs(np.inner(values, h) + values.T @ J @ values))
    factor = np.std(results, ddof=1) / np.sqrt(len(h))
    print(factor)
    return h / factor, J / factor


# This function generates all n bit Gray 
# codes and prints the generated codes
def generateGrayarr(n):
    # base case
    if (n <= 0):
        return
 
    # 'arr' will store all generated codes
    arr = list()
 
    # start with one-bit pattern
    arr.append("0")
    arr.append("1")
 
    # Every iteration of this loop generates 
    # 2*i codes from previously generated i codes.
    i = 2
    j = 0
    while(True):
        if i >= 1 << n:
            break
        # Enter the previously generated codes 
        # again in arr[] in reverse order. 
        # Nor arr[] has double number of codes.
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])
 
        # append 0 to the first half
        for j in range(i):
            arr[j] = "0" + arr[j]
 
        # append 1 to the second half
        for j in range(i, 2 * i):
            arr[j] = "1" + arr[j]
        i = i << 1

    return arr

def plot(n, m, D):
    d = MatrixSymbol("d", n, n)
    y = MatrixSymbol("y", n, 2)
    r = MatrixSymbol("r", int(n*(n-1)/2), 1) # only up-right 
    res = H_score_onelocal(n, m, D, d, y, r)
    res = simplify(expand(res))
    
    gray = generateGrayarr(n * (n-1))
    for x in gray:
        res2 = res.copy()
        for i, j in enumerate(x):
            j = 0 if j == '0' else 1
            res2 = res2.subs({num_to_symbol(i, n, d, y, r): j})
        print(x, float(res2))

def bf(C, h, J):
    values = np.zeros((len(h),))
    bf_results = {}
    for x in tqdm(range(1 << len(h))):
        orig_x = x
        for i in range(len(h)):
            values[i] = -1 if x & 1 == 0 else 1
            x >>= 1
        value = C + np.inner(h, values) + values.T @ J @ values
        bf_results["{:b}".format(orig_x)] = float(value)
    print("Brute force results:", dict(heapq.nsmallest(10, bf_results.items(), key=itemgetter(1))))

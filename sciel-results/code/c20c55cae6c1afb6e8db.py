import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate powerful numbers up to L
    L = 10**8
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = (a * a) * b3
            if val <= L:
                powerfuls.add(val)
    
    P = np.array(sorted(powerfuls), dtype=np.int64)
    n_p = len(P)
    
    if n_p == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Count sums using NumPy
    counts = {}
    best_N = 0
    best_count = 0
    
    for i in range(n_p):
        if time.time() > deadline - 10:
            break
        a = P[i]
        # Vectorized: all sums a + P[j] for j >= i
        sums = a + P[i:]
        # Count occurrences
        unique, cnts = np.unique(sums, return_counts=True)
        for k in range(len(unique)):
            s = int(unique[k])
            c = int(cnts[k])
            old = counts.get(s, 0)
            new_c = old + c
            counts[s] = new_c
            if new_c > best_count:
                best_count = new_c
                best_N = s
    
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Construct pairs
    P_list = P.tolist()
    powerfuls_set = set(P_list)
    pairs = []
    for a in P_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}

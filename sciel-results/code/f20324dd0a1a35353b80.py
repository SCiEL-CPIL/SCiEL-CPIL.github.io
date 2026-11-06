import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    limit = 10000000000 
    
    pows = set([1])
    max_i = math.isqrt(limit)
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    for i in range(1, max_i + 1):
        sq = i * i
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
        for j in range(1, max_j + 1):
            val = sq * cubes[j-1]
            if val <= limit:
                pows.add(val)
            else:
                break
        if time.time() > deadline - 50:
             break

    P = np.array(sorted(list(pows)), dtype=np.int64)
    K = len(P)
    
    best_count_full = 0
    best_sum = 0
    
    block_size = 500
    for start in range(0, K, block_size):
        end = min(start + block_size, K)
        block = P[start:end]
        
        sums = block[:, None] + P[None, :]
        flat_sums = sums.flatten()
        
        unique_sums, counts = np.unique(flat_sums, return_counts=True)
        
        max_c = np.max(counts)
        if max_c > best_count_full:
            best_count_full = max_c
            idx = np.argmax(counts)
            best_sum = unique_sums[idx]
            
        if time.time() > deadline - 10:
            break
            
    N_diag = 0
    if best_sum % 2 == 0:
        half = best_sum // 2
        idx_half = np.searchsorted(P, half)
        if idx_half < K and P[idx_half] == half:
            N_diag = 1
            
    actual_best_count = (best_count_full + N_diag) // 2
    
    pairs = []
    pows_set = set(P.tolist())
    for a in P:
        b = best_sum - a
        if b < a:
            break
        if b in pows_set:
            pairs.append([int(a), int(b)])
            
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': int(best_sum), 'pairs': pairs}

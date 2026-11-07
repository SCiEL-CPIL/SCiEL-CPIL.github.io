import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = 80.0  # 10s margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 20:
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline - 10:
        return [11, 6]

    # 2. Group indices by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    if time.time() - start_time > deadline - 5:
        return [11, 6]

    # 3. Search for best pair
    best_num = 4  # |2*11 - 3*6| = 4
    best_den = 12  # 2*6
    best_a = 11
    best_b = 6

    for s, indices in groups.items():
        if len(indices) < 2:
            continue
            
        indices.sort()
        n_idx = len(indices)
        
        for bi in range(n_idx):
            b = indices[bi]
            target = 3 * b // 2  # a ≈ 1.5*b
            
            pos = bisect_left(indices, target)
            
            # Check a wider range of candidates around pos
            for offset in range(-5, 6):
                idx = pos + offset
                if 0 <= idx < n_idx:
                    a = indices[idx]
                    if a == b:
                        continue
                    
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    den = 2 * b
                    
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a
                        best_b = b

    return [int(best_a), int(best_b)]

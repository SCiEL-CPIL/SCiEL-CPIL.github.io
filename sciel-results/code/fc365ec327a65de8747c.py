import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 10:  # Reserve 10s for search
            return [11, 6]  # sigma(6)=12, sigma(11)=12
        sigma[i::i] += i
    
    if time.time() - start_time > deadline - 5:
        return [11, 6]
    
    # 2. Search for pairs where a/b ≈ 3/2 and sigma(a) = sigma(b)
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    for b in range(1, N + 1):
        if time.time() - start_time > deadline:
            break
        
        target = 1.5 * b
        a_floor = int(math.floor(target))
        a_ceil = int(math.ceil(target))
        
        candidates = []
        if a_floor >= 1 and a_floor <= N:
            candidates.append(a_floor)
        if a_ceil >= 1 and a_ceil <= N and a_ceil != a_floor:
            candidates.append(a_ceil)
        
        for a in candidates:
            if a == b:
                continue
            if 2 * a == 3 * b:
                # Exact 3/2 ratio is invalid
                continue
            if sigma[a] == sigma[b]:
                numerator = abs(2 * a - 3 * b)
                denominator = 2 * b
                err = numerator / denominator
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
    
    return [int(best_a), int(best_b)]

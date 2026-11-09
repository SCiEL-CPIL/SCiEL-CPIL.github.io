import numpy as np
import time
from bisect import bisect_left

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Compute sigma(n) for all n <= N
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]

    # 2. Build dictionary: sigma value -> sorted list of indices
    sigma_to_indices = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in sigma_to_indices:
            sigma_to_indices[s] = []
        sigma_to_indices[s].append(n)
    
    # 3. Search for best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    for b in range(N, 0, -1):
        if 1.0 / (2.0 * b) > best_error:
            break
        if time.time() > deadline:
            break
            
        sb = int(sigma[b])
        candidates = sigma_to_indices.get(sb)
        if not candidates:
            continue
        
        target = 1.5 * b
        pos = bisect_left(candidates, int(target))
        
        # Check a window around pos
        lo = max(0, pos - 30)
        hi = min(len(candidates), pos + 31)
        for idx in range(lo, hi):
            a = candidates[idx]
            if a == b:
                continue
            D = abs(2 * a - 3 * b)
            if D == 0:
                continue
            error = D / (2.0 * b)
            if error < best_error:
                best_error = error
                best_a = a
                best_b = b
    
    return [int(best_a), int(best_b)]

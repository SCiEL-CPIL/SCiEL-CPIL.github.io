import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24
    
    S = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            if time.time() > deadline - 10:
                break
            S[pk::pk] *= p
            pk *= p
        if time.time() > deadline - 10:
            break
    
    if time.time() > deadline - 10:
        return min_n

    total_collisions = np.zeros(max_val + 1, dtype=np.int32)
    
    for delta in range(1, 24):
        M = (S[:-delta] == S[delta:]).astype(np.int32)
        W = 24 - delta
        
        if W <= 0:
            continue
        
        C = np.cumsum(M, dtype=np.int32)
        K = C[W:] - C[:-W]
        
        start_idx = 1000
        end_idx = 1000001
        k_start = 1001
        k_end = 1000002
        
        if len(K) >= k_end:
            total_collisions[start_idx:end_idx] += K[k_start:k_end]
            
        if time.time() > deadline - 5:
            break
            
    search_range = total_collisions[1000:1000001]
    max_collisions = np.max(search_range)
    best_offset = np.argmax(search_range)
    best_n = 1000 + best_offset
    
    return best_n


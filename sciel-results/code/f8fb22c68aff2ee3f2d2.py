import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24
    
    small_part = np.ones(max_val + 1, dtype=np.int64)
    idx = np.arange(1, max_val + 1)
    
    for p in primes:
        power = p
        while power <= max_val:
            mask = (idx % power == 0)
            if np.any(mask):
                small_part[mask] *= p
            power *= p
            if power > max_val:
                break
    
    sp = small_part.tolist()
    
    best_n = min_n
    best_count = 25
    
    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
        window = sp[n+1:n+25]
        distinct = len(set(window))
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 2:
                break
    
    return best_n

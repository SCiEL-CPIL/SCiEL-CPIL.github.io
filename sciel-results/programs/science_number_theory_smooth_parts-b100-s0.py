import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted factors using NumPy
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        # For multiples of p, compute the p-part
        multiples = np.arange(p, MAX_VAL + 1, p)
        # For each multiple, divide out p until not divisible
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts
    
    # Convert to list for faster set operations
    extracted_list = extracted.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        factors = set()
        for i in range(1, WINDOW + 1):
            factors.add(extracted_list[n + i])
        count = len(factors)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
    
    return best_n

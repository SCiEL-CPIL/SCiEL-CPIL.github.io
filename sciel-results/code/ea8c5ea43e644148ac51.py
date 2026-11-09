import numpy as np
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range: n in [1000, 1000000], so n+i in [1001, 1000024]
    lo, hi = 1001, 1000025  # hi exclusive
    
    # Precompute the small-prime factor for each number in [lo, hi)
    factor = np.arange(lo, hi, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p in range, divide out all factors of p
        for val in range(lo, hi, p):
            idx = val - lo
            while factor[idx] % p == 0:
                factor[idx] //= p
    
    # Map factor values to integer IDs for efficient counting
    unique_factors, inverse = np.unique(factor, return_inverse=True)
    num_unique = len(unique_factors)
    
    # Sliding window with incremental distinct count
    counts = np.zeros(num_unique, dtype=np.int32)
    distinct = 0
    
    # Initialize first window (positions 0..23)
    for j in range(24):
        v = inverse[j]
        if counts[v] == 0:
            distinct += 1
        counts[v] += 1
    
    best_score = distinct
    best_n = 1000  # i=0 corresponds to n=1000
    
    n_windows = len(factor) - 24
    
    for i in range(1, n_windows):
        # Remove element at position i-1
        v_out = inverse[i - 1]
        counts[v_out] -= 1
        if counts[v_out] == 0:
            distinct -= 1
        
        # Add element at position i+23
        v_in = inverse[i + 23]
        if counts[v_in] == 0:
            distinct += 1
        counts[v_in] += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = 1000 + i
            
            if best_score <= 1:
                break
        
        if time.time() > deadline:
            break
    
    return best_n

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts for all numbers up to 1000024
    # smooth_part[m] = the factor of m composed of primes < 24
    max_val = 1000024
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, compute the p-part
        multiples = np.arange(p, max_val + 1, p)
        # Compute p-part for each multiple using vectorized operations
        temp = multiples.copy()
        factor = np.ones(len(multiples), dtype=np.int64)
        while np.any(temp % p == 0):
            mask = (temp % p == 0)
            factor[mask] *= p
            temp[mask] //= p
        smooth_part[multiples] *= factor
    
    # Now scan for best n: minimize distinct smooth parts in window of 24
    best_n = 1000
    best_count = 25  # worst case is 24 distinct + possibly 1 more
    
    # Use numpy array for faster access
    sp = smooth_part
    
    for n in range(1000, 1000000):
        if time.time() > deadline:
            break
        # Extract window and count distinct values
        window = sp[n+1:n+25]
        c = len(np.unique(window))
        if c < best_count:
            best_count = c
            best_n = n
            if best_count <= 2:
                break
    
    return best_n
# EVOLVE-BLOCK-END

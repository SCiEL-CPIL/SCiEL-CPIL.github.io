"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth parts for all numbers up to 1000024
    max_n = 1000000
    max_val = max_n + 25  # We need n+24 where n up to 1000000
    
    # smooth_part[m] = factor of m composed of primes < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            m = multiple
            while m % p == 0:
                smooth_part[multiple] *= p
                m //= p
    
    # Convert to Python list for faster access in tight loop
    sp = smooth_part.tolist()
    
    best_n = 1000
    best_score = 25  # worst case
    
    # Use sliding window with frequency count for O(1) distinct updates
    # Window for n: sp[n+1] to sp[n+24] (24 elements)
    
    from collections import defaultdict
    freq = defaultdict(int)
    distinct_count = 0
    
    # First window for n=1000: sp[1001] to sp[1024]
    for i in range(1001, 1025):
        val = sp[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
    
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
    
    # Slide: from n to n+1, remove sp[n+1], add sp[n+25]
    n = 1000
    while n < max_n:
        if time.time() > deadline:
            break
        
        # Remove sp[n+1]
        old_val = sp[n + 1]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
        
        # Add sp[n+25]
        new_val = sp[n + 25]
        if freq[new_val] == 0:
            distinct_count += 1
        freq[new_val] += 1
        
        n += 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score <= 2:
                pass
    
    return best_n
# EVOLVE-BLOCK-END

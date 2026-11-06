"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(small_primes)
    
    # Range: n in [1000, 1000000], so n+1 in [1001, 1000001], n+24 in [1024, 1000024]
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute smooth part for all numbers up to max_val
    # smooth_part[m] = the part of m composed only of primes < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    smooth_part[0] = 0
    
    for p in small_primes:
        max_power = 1
        while p ** max_power <= max_val:
            max_power += 1
        max_power -= 1
        
        for k in range(1, max_power + 1):
            p_k = p ** k
            indices = np.arange(p_k, max_val + 1, p_k)
            smooth_part[indices] *= p
            
    # Now slide window: for each n in [1000, 1000000], look at n+1..n+24
    # Count distinct smooth parts
    best_n = 1000
    best_count = float('inf')
    
    # Use a dictionary for counts to avoid Counter overhead
    counts = {}
    
    # Initial window: n=1000, numbers 1001..1024
    for m in range(1001, 1025):
        sp = smooth_part[m]
        counts[sp] = counts.get(sp, 0) + 1
        
    distinct_count = len(counts)
    best_count = distinct_count
    best_n = 1000
    
    if best_count <= 2:
        return best_n
    
    n_start = 1001
    n_end = max_n
    
    for n in range(n_start, n_end + 1):
        # Remove n (the leftmost), add n+24 (the rightmost)
        out_val = n
        in_val = n + 24
        sp_out = smooth_part[out_val]
        sp_in = smooth_part[in_val]
        
        if sp_out != sp_in:
            # Update out_val
            c_out = counts[sp_out]
            if c_out == 1:
                del counts[sp_out]
                distinct_count -= 1
            else:
                counts[sp_out] = c_out - 1
                
            # Update in_val
            c_in = counts.get(sp_in, 0)
            if c_in == 0:
                distinct_count += 1
            counts[sp_in] = c_in + 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 2:
                break
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END

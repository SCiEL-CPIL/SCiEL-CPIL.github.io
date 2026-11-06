"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    """
    Find n in [1000, 1000000] such that for each n+i (1<=i<=24),
    we extract the smooth part (product of prime powers for primes < 24),
    and minimize the number of DISTINCT smooth parts across all 24 values.
    
    Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    
    Strategy: Precompute the smooth part for all numbers up to 1000000+24,
    then slide a window of 24 and count distinct smooth parts.
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute smooth part for each number
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        # For each multiple of p, divide out all factors of p and track
        # the smooth contribution
        # Actually, we want: smooth_part[x] = product of p^k for each prime p<24
        # where p^k is the highest power of p dividing x
        # So smooth[x] = x / (x with all small prime factors removed)
        # Equivalently: smooth[x] = x * prod_{p<24} (1/p^v_p(x))... no
        # smooth[x] = prod_{p<24} p^{v_p(x)}
        
        # Start with all 1s, multiply in p^v_p(x) for each x
        for multiple in range(p, max_val + 1, p):
            power = 1
            temp = multiple
            while temp % p == 0:
                power *= p
                temp //= p
            smooth[multiple] *= power
    
    # Now slide window: for n in [1000, 1000000], look at smooth[n+1]..smooth[n+24]
    # Count distinct values in that window
    best_n = 1000
    best_count = float('inf')
    
    # Use a set for each window - but that's slow in pure Python
    # Let's use numpy unique
    n_start = 1000
    n_end = 1000000
    
    # Process in chunks to check deadline
    chunk_size = 50000
    for n0 in range(n_start, n_end + 1, chunk_size):
        n1 = min(n0 + chunk_size, n_end + 1)
        # For each n in [n0, n1-1], window is smooth[n+1 : n+25]
        # Vectorize: create array of all smooth values in range [n0+1, n1+24]
        # For n, the window indices are n+1 to n+24
        
        # Simple loop within chunk
        for n in range(n0, n1):
            window = smooth[n+1:n+25]
            distinct = len(np.unique(window))
            if distinct < best_count:
                best_count = distinct
                best_n = n
                if best_count == 1:
                    break  # Can't do better than 1
        if time.time() > deadline:
            break
        if best_count == 1:
            break
    
    return best_n
# EVOLVE-BLOCK-END

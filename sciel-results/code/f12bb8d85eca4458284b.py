"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range: n in [1000, 1000000], so n+1 in [1001, 1000001], n+24 in [1024, 1000024]
    # We need smooth parts for numbers from 1001 to 1000024
    max_val = 1000024
    
    # Precompute 23-smooth part for all numbers up to max_val using vectorized ops
    # smooth_part[m] = the part of m composed only of primes < 24
    smooth_part = np.zeros(max_val + 1, dtype=np.int64)
    smooth_part[1:] = 1  # Start with 1 for all, will multiply by prime factors
    
    for p in small_primes:
        # For each multiple of p, extract the full p-power factor
        # We need to find the highest power of p dividing each number
        # Use repeated division approach but vectorized
        multiples = np.arange(p, max_val + 1, p)
        # For each multiple, compute the p-part
        # Start with 1, multiply by p for each factor
        p_parts = np.ones(len(multiples), dtype=np.int64)
        remaining = multiples.copy()
        while True:
            mask = (remaining % p == 0)
            if not np.any(mask):
                break
            p_parts[mask] *= p
            remaining[mask] //= p
        # Now we need to assign these p_parts to the correct positions
        # But we need to multiply into smooth_part, not just set
        # So we need to track which numbers get which p-part
        # Actually, let's do it differently: for each number, accumulate the p-part
        # This is tricky with vectorization because different numbers have different p-parts
        
        # Alternative: use a loop over primes but vectorize the inner while
        # Let me reconsider: for each prime p, for each multiple of p,
        # we want to multiply smooth_part[m] by p^k where k is the exponent of p in m
        # We can do this by iterating over powers of p
        
        # For p^1: all multiples of p get factor p
        # For p^2: all multiples of p^2 get another factor p
        # etc.
        power = p
        while power <= max_val:
            multiples_power = np.arange(power, max_val + 1, power)
            smooth_part[multiples_power] *= p
            power *= p
            if power > max_val:
                break
    
    # Now slide window: for each n in [1000, 1000000], look at n+1..n+24
    # Count distinct smooth parts
    best_n = 1000
    best_count = float('inf')
    
    n_start = 1000
    n_end = 1000000
    
    # Initial window: n=1000, numbers 1001..1024
    window_indices = np.arange(n_start + 1, n_start + 25)
    window_vals = smooth_part[window_indices]
    unique_vals, counts_arr = np.unique(window_vals, return_counts=True)
    counts = dict(zip(unique_vals, counts_arr))
    best_count = len(counts)
    best_n = n_start
    
    for n in range(n_start + 1, n_end + 1):
        # Remove n (the leftmost), add n+24 (the rightmost)
        out_val = n
        in_val = n + 24
        sp_out = smooth_part[out_val]
        sp_in = smooth_part[in_val]
        
        counts[sp_out] -= 1
        if counts[sp_out] == 0:
            del counts[sp_out]
        counts[sp_in] = counts.get(sp_in, 0) + 1
        
        current_count = len(counts)
        if current_count < best_count:
            best_count = current_count
            best_n = n
            if best_count <= 2:
                break  # Can't do much better than 1 or 2
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END

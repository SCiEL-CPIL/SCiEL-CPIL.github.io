"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(small_primes)
    
    # Range: n in [1000, 1000000], so n+1 in [1001, 1000001], n+24 in [1024, 1000024]
    max_val = 1000024
    
    # Precompute the 23-smooth part (part composed only of primes < 24) for all numbers up to max_val
    # smooth_part[m] = the largest divisor of m that is 23-smooth
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            # For each multiple of power, multiply smooth_part by p
            multiples_power = np.arange(power, max_val + 1, power)
            smooth_part[multiples_power] *= p
            power *= p
    
    n_start = 1000
    n_end = 1000000
    
    # Use a bitmask approach: for each number, compute which small primes divide its smooth part
    # Actually, we need the number of DISTINCT smooth parts in the window [n+1, n+24]
    # The smooth part is a specific integer value, not just a set of primes
    # So we need to count distinct smooth_part values in the window
    
    # Precompute smooth_part for all values in the range
    # We already have smooth_part array
    
    # Sliding window with frequency counting
    # For the window [n+1, n+24], we want the number of distinct smooth_part values
    
    # Initialize window for n = n_start
    window_indices = np.arange(n_start + 1, n_start + 25)
    window_vals = smooth_part[window_indices]
    
    # Count frequencies
    from collections import defaultdict
    counts = defaultdict(int)
    for v in window_vals:
        counts[v] += 1
    
    best_count = len(counts)
    best_n = n_start
    
    # Slide the window
    for n in range(n_start + 1, n_end + 1):
        # Remove n (leftmost element leaving the window)
        out_val = n
        # Add n+24 (rightmost element entering the window)
        in_val = n + 24
        
        sp_out = smooth_part[out_val]
        sp_in = smooth_part[in_val]
        
        # Decrement count for sp_out
        counts[sp_out] -= 1
        if counts[sp_out] == 0:
            del counts[sp_out]
        
        # Increment count for sp_in
        counts[sp_in] += 1
        
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

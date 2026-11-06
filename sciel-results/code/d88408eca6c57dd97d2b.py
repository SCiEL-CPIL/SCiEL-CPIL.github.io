import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024
    
    # Initialize rough_part with m
    rough_part = np.arange(max_val + 1, dtype=np.int64)
    
    # Sieve: remove small prime factors from rough_part
    for p in small_primes:
        # Get indices of multiples of p
        multiples = np.arange(p, max_val + 1, p)
        # Extract current values
        vals = rough_part[multiples].copy()
        
        # Divide out p as many times as possible
        # We need to do this iteratively because vectorized division 
        # only removes one factor at a time if we just do vals //= p
        # But we can check divisibility.
        
        # Optimization: 
        # Instead of a while loop with np.any, we can use a loop that 
        # checks if any are still divisible.
        # The max power of 2 in 10^6 is ~20. 
        # So at most 20 iterations for p=2, 1 for p=23.
        
        while True:
            mask = vals % p == 0
            if not np.any(mask):
                break
            vals[mask] //= p
        
        rough_part[multiples] = vals
    
    # smooth_part[m] = m // rough_part[m]
    smooth_part = np.zeros(max_val + 1, dtype=np.int64)
    smooth_part[1:] = np.arange(1, max_val + 1) // rough_part[1:]
    
    # Search for best n
    best_n = 1000
    best_count = 25  # max distinct is 24, so 25 is safe upper bound
    
    # We can speed up the search by checking every n.
    # 1e6 iterations of set(window) is fast in Python if window is small.
    # But let's try to be clever.
    
    # If we find a count of 1, we stop.
    # If we find a count of 2, we keep looking for 1.
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        window = smooth_part[n+1:n+25]
        # Using set on a numpy array slice
        distinct_count = len(set(window))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break
                
    return best_n

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s
    
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # we need smooth parts up to 1000024
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute smooth part for each number using sieve-like approach
    # smooth_part[i] = largest divisor of i with only prime factors in small_primes
    smooth = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        for multiple in range(p, MAX_VAL + 1, p):
            # Multiply smooth[multiple] by p as long as p divides multiple
            # Actually, we need to accumulate all small prime factors
            pass
    
    # Better approach: for each number, compute its smooth part directly
    # Use a sieve to accumulate
    smooth = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        for multiple in range(p, MAX_VAL + 1, p):
            # Count how many times p divides multiple
            temp = multiple
            while temp % p == 0:
                smooth[multiple] *= p
                temp //= p
    
    # Now find the window with minimum distinct smooth parts
    best_n = 1000
    best_count = 999999
    
    # Use a sliding window with a frequency counter
    # For efficiency, use numpy where possible
    
    # First pass: just scan all windows
    for n in range(1000, MAX_N + 1):
        if time.time() - start_time > deadline:
            break
        window_smooth = smooth[n+1:n+1+WINDOW]
        distinct_count = len(np.unique(window_smooth))
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count <= 2:
                break
    
    return best_n

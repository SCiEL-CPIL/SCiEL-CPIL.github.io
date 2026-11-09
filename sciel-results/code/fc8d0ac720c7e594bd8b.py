import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    
    # Sieve for Largest Prime Factor using NumPy
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(LIMIT + 1, dtype=np.int32)
    
    for i in range(2, LIMIT + 1):
        if lpf[i] == 0:
            # i is prime
            # Mark all multiples of i with i
            # Using slice assignment
            lpf[i::i] = i
            
            if time.time() > deadline:
                return 10000

    # Scan for the optimal n
    # We need max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 10001 to 1000001
    
    # Create arrays for the range
    # Note: lpf is 0-indexed.
    # n ranges from 10000 to MAX_N.
    # lpf[n] is lpf[10000:MAX_N+1]
    # lpf[n+1] is lpf[10001:MAX_N+2]
    
    if time.time() > deadline:
        return 10000
        
    lpf_n = lpf[10000:MAX_N + 1]
    lpf_np1 = lpf[10001:MAX_N + 2]
    
    # Compute the max for each pair
    current_vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(current_vals)
    
    best_n = 10000 + min_idx
    
    return int(best_n)

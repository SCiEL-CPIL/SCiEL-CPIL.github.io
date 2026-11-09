import numpy as np
import time

def construct():
    start_time = time.time()
    
    # We need to calculate LPF for n and n+1
    # Max n is 1,000,000, so max n+1 is 1,000,001
    max_val = 1000001
    size = max_val + 1
    
    # Initialize LPF array with 0
    lpf = np.zeros(size, dtype=np.int64)
    
    # Sieve for Largest Prime Factor
    # Iterate through numbers from 2 to max_val
    # If lpf[i] == 0, i is prime
    # Update multiples of i with i
    for i in range(2, size):
        if lpf[i] == 0:
            # i is prime, update all multiples
            lpf[i::i] = i
            
    # Calculate max LPF for consecutive pairs n, n+1
    # n ranges from 10000 to 1000000
    # lpf[n] corresponds to index n
    # lpf[n+1] corresponds to index n+1
    # We want to find n in [10000, 1000000]
    
    # Extract slices
    # n is 10000..1000000
    # n+1 is 10001..1000001
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    # Compute element-wise maximum
    max_lpf_pairs = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum value
    # If there are multiple, np.argmin returns the first one
    min_idx = np.argmin(max_lpf_pairs)
    
    # The actual n value
    best_n = 10000 + min_idx
    
    return int(best_n)

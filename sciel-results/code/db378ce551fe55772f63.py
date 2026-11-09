import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    max_val = 1000001
    size = max_val + 1
    
    # Use int32 for better memory locality and speed
    lpf = np.zeros(size, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # We iterate i from 2 to size-1
    # If lpf[i] is 0, i is prime
    for i in range(2, size):
        if lpf[i] == 0:
            # i is prime, update all multiples
            # lpf[i::i] = i
            # This is vectorized and fast
            lpf[i::i] = i
            
    if time.time() - start_time > deadline:
        # Fallback if somehow slow (should not happen)
        # Return a known good candidate or just the first one
        return 10000

    # n ranges from 10000 to 1000000
    # lpf[n] is the LPF of n
    # lpf[n+1] is the LPF of n+1
    
    # Extract slices
    # n: 10000 to 1000000 (inclusive) -> indices 10000 to 1000000
    # n+1: 10001 to 1000001 (inclusive) -> indices 10001 to 1000001
    
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    # Compute element-wise maximum
    max_lpf_pairs = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf_pairs)
    
    # The actual n value
    best_n = 10000 + min_idx
    
    return int(best_n)

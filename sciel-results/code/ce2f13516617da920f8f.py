import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 50  # Safety margin for 90s limit
    
    N = 1000000
    max_val = N + 1  # We need lpf for N+1 as well
    
    # Sieve to find Largest Prime Factor using NumPy
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    
    # Handle 2 separately
    lpf[2::2] = 2
    
    # Only check odd numbers
    for i in range(3, max_val + 1, 2):
        if lpf[i] == 0:
            # i is prime, mark all odd multiples
            lpf[i::2*i] = i
    
    # Check time
    if time.time() - start_time > deadline:
        return 10000  # Fallback, unlikely to happen
        
    min_lpf = float('inf')
    best_n = 10000
    
    # Vectorized scan for minimum max(lpf[n], lpf[n+1])
    n_range = np.arange(10000, N + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    min_idx = np.argmin(max_lpf)
    min_lpf = max_lpf[min_idx]
    best_n = int(n_range[min_idx])
    
    return best_n

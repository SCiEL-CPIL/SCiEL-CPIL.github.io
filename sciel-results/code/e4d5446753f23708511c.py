import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    
    max_val = N_MAX + 1
    
    # Use NumPy for faster sieve
    lpf_arr = np.ones(max_val + 1, dtype=np.int32)
    
    for i in range(2, max_val + 1):
        if lpf_arr[i] == 1:
            # Mark all multiples of i with i as largest prime factor
            indices = np.arange(i, max_val + 1, i)
            lpf_arr[indices] = i
    
    # Vectorized scan
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf_arr[n_range]
    lpf_np1 = lpf_arr[n_range + 1]
    combined_lpf = np.maximum(lpf_n, lpf_np1)
    
    min_idx = np.argmin(combined_lpf)
    best_lpf = combined_lpf[min_idx]
    best_n = n_range[min_idx]
    
    return int(best_n)

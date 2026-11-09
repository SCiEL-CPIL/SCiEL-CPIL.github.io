import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 55  # 5s margin
    
    N = 1000000
    max_val = N + 1
    
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
            
    if time.time() - start_time > deadline:
        return 10000
        
    # Extract ranges
    # n is in [10000, 1000000]
    # lpf[n] is lpf[10000:1000001]
    # lpf[n+1] is lpf[10001:1000002]
    
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    min_idx = np.argmin(max_lpf)
    best_n = 10000 + min_idx
    
    return int(best_n)

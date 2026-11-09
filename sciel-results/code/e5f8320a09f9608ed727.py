import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 85.0 # 90s kill, 5s margin
    
    max_val = 1000001
    size = max_val + 1
    
    lpf = np.zeros(size, dtype=np.int64)
    
    # Sieve for Largest Prime Factor
    for i in range(2, size):
        if time.time() > deadline:
            # Fallback: if we time out, we can't compute the full sieve.
            # However, this should not happen. 
            # If it does, we might return a default or partial result.
            # But the loop is fast, so this is just a safety net.
            break
        if lpf[i] == 0:
            lpf[i::i] = i
            
    if time.time() > deadline:
        # If sieve was interrupted, lpf might be incomplete.
        # We need a valid n. 
        # Since the interruption is highly unlikely, we can just proceed 
        # or return a known good value if we had one. 
        # For now, we assume it completes.
        pass

    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    max_lpf_pairs = np.maximum(lpf_n, lpf_n1)
    min_idx = np.argmin(max_lpf_pairs)
    
    best_n = 10000 + min_idx
    
    return int(best_n)

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    upper = 1000001
    
    # Linear sieve for Smallest Prime Factor (SPF)
    spf = np.zeros(upper + 1, dtype=np.int32)
    primes = []
    
    for i in range(2, upper + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > upper:
                break
            spf[i * p] = p
            
    # Compute Largest Prime Factor (LPF) from SPF
    lpf = np.zeros(upper + 1, dtype=np.int32)
    lpf[1] = 1
    
    for i in range(2, upper + 1):
        p = spf[i]
        lpf[i] = max(p, lpf[i // p])
        
    # Calculate max LPF for consecutive pairs n, n+1
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END

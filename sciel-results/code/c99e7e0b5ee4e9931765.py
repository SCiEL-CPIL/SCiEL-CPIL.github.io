import numpy as np

def construct():
    # We need to find n in [10000, 1000000] that minimizes
    # max(lpf(n), lpf(n+1))
    
    upper = 1000001
    
    # Sieve for Largest Prime Factor
    # lpf[i] stores the largest prime factor of i
    lpf = np.zeros(upper + 1, dtype=np.int32)
    
    # For each i from 2 to upper, if lpf[i] is 0, i is prime.
    # Assign i to all multiples of i.
    # Since we iterate in increasing order, the last assignment is by the largest prime factor.
    for i in range(2, upper + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
            
    # n ranges from 10000 to 1000000
    # lpf(n) is lpf[10000] to lpf[1000000]
    # lpf(n+1) is lpf[10001] to lpf[1000001]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Compute max LPF for each consecutive pair
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the n with the minimum max LPF
    min_idx = np.argmin(max_lpf)
    best_n = 10000 + min_idx
    
    return int(best_n)

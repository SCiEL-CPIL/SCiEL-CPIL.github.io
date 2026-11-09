import numpy as np

def construct():
    N = 123457
    
    lpf = np.zeros(N + 1, dtype=np.int64)
    lpf[1] = 1
    
    # Sieve up to sqrt(N)
    limit = int(np.sqrt(N))
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
            
    # Handle primes > sqrt(N)
    # These are numbers i > limit such that lpf[i] == 0
    # They are prime, so their largest prime factor is themselves
    indices = np.arange(N + 1)
    mask = (indices > limit) & (lpf == 0)
    lpf[mask] = indices[mask]
    
    # Scan
    lpf_a = lpf[1:N]
    lpf_b = lpf[N-1:0:-1]
    max_pfs = np.maximum(lpf_a, lpf_b)
    
    min_max_pf = np.min(max_pfs)
    best_idx = np.argmin(max_pfs)
    
    best_a = best_idx + 1
    best_b = N - best_a
    
    return [best_a, best_b]

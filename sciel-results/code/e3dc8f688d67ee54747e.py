"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    # Compute largest prime factor for all numbers 1..N using a sieve.
    # lpf[n] = largest prime factor of n. For n=1, lpf[1]=1.
    lpf = np.ones(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            lpf[i::i] = i
    
    # Find a in [1, N-1] that minimizes max(P(a), P(N-a))
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    pa = lpf[a]
    pb = lpf[b]
    m = np.maximum(pa, pb)
    idx = int(np.argmin(m))
    
    best_a = int(a[idx])
    best_b = int(b[idx])
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END

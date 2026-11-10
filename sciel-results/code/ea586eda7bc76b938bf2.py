import numpy as np
import math

def construct():
    N = 1000000
    NMAX = N + 1
    
    # Sieve of Eratosthenes using NumPy
    is_prime = np.ones(NMAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(NMAX)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Compute Largest Prime Factor
    lpf = np.zeros(NMAX + 1, dtype=np.int32)
    primes = np.where(is_prime)[0]
    for p in primes:
        lpf[p::p] = p
        
    # Find the n in [10000, 1000000] that minimizes max(LPF(n), LPF(n+1))
    # lpf[n] for n in 10000..1000000
    # lpf[n+1] for n in 10000..1000000 -> indices 10001..1000001
    curr_lpf = lpf[10000:N+1]
    next_lpf = lpf[10001:N+2]
    
    max_lpf = np.maximum(curr_lpf, next_lpf)
    min_val = np.min(max_lpf)
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    
    return int(best_n)

import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000
    M = N_MAX + 1  # We need lpf for values up to 1000001

    # Sieve of Eratosthenes using numpy
    is_prime = np.ones(M + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    limit = math.isqrt(M) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False

    # Collect primes
    primes = np.where(is_prime)[0]
    
    # Compute largest prime factor for each number up to M
    # lpf[x] = largest prime factor of x
    # Use int32 to save memory and improve cache performance
    lpf = np.zeros(M + 1, dtype=np.int32)
    
    # Iterate through primes. Since we process in increasing order, 
    # the last prime that divides a number is the largest prime factor.
    # lpf[multiple] = p overwrites previous values, so the largest p wins.
    for p in primes:
        p_int = int(p)
        # Generate multiples of p
        multiples = np.arange(p_int, M + 1, p_int, dtype=np.int32)
        lpf[multiples] = p_int
        
    # Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    
    # Vectorized search
    lpf_n = lpf[N_MIN : N_MAX + 1]
    lpf_np1 = lpf[N_MIN + 1 : N_MAX + 2]
    
    # Compute max for each pair
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    best_n = N_MIN + min_idx
    best_val = max_lpf[min_idx]
    
    return int(best_n)

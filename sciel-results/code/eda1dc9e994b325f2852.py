"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    N_MAX = 1000001  # n+1 when n=1000000
    N_MIN = 10000
    
    # Precompute primes up to sqrt(N_MAX) ≈ 1000
    limit = int(np.sqrt(N_MAX)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Compute LPF for all numbers in [N_MIN, N_MAX]
    # lpf[i] stores the largest prime factor of (N_MIN + i)
    size = N_MAX - N_MIN + 1
    lpf = np.zeros(size, dtype=np.int32)
    
    for p in primes:
        if p > N_MAX:
            break
        # Find first multiple of p in [N_MIN, N_MAX]
        start = ((N_MIN + p - 1) // p) * p
        for j in range(start - N_MIN, size, p):
            lpf[j] = p
    
    # For numbers that are prime themselves (larger than sqrt(N_MAX)),
    # their LPF is themselves. Check if lpf[i] == 0, meaning no prime factor found
    # that means the number is prime
    for i in range(size):
        if lpf[i] == 0:
            val = N_MIN + i
            if val >= 2:
                lpf[i] = val
    
    # Compute max(lpf[n], lpf[n+1]) for n in [N_MIN, 1000000]
    # n ranges from 10000 to 1000000, so n+1 ranges from 10001 to 1000001
    # In our array: index for n is (n - N_MIN), index for n+1 is (n+1 - N_MIN)
    n_vals = np.arange(N_MIN, 1000001)
    idx_n = n_vals - N_MIN
    idx_n1 = idx_n + 1
    
    max_lpf = np.maximum(lpf[idx_n], lpf[idx_n1])
    
    # Find the index with minimum max_lpf
    min_idx = np.argmin(max_lpf)
    best_n = n_vals[min_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1
    
    # 1. Sieve of Eratosthenes to find all primes up to limit
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = b'\x00' * len(is_prime[i*i : limit+1 : i])
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # 2. Compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        for multiple in range(p, limit + 1, p):
            lpf[multiple] = p
    
    # 3. Search for n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    # Use numpy for vectorized computation
    n_values = np.arange(N_MIN, N_MAX + 1)
    scores = np.maximum(lpf[n_values], lpf[n_values + 1])
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    best_n = int(n_values[min_idx])
    best_score = int(scores[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END

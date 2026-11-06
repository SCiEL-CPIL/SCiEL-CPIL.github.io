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
    
    # 1. Sieve to compute Largest Prime Factor (LPF) for each number up to limit
    # Using NumPy for speed
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            # Update all multiples of i
            lpf[i::i] = i
    
    # 2. Find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    # Vectorized computation
    n_range = np.arange(N_MIN, N_MAX + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
# EVOLVE-BLOCK-END

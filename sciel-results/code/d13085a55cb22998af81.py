"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # we need lpf up to 1000001
    
    # Sieve for largest prime factor
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    # Vectorized scan
    n_range = np.arange(10000, MAX_N + 1)
    vals = np.maximum(lpf[10000:MAX_N + 1], lpf[10001:MAX_N + 2])
    
    min_idx = np.argmin(vals)
    best_n = 10000 + min_idx
    best_val = vals[min_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END

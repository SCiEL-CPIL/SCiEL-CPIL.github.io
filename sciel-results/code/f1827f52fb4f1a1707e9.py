import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 1000000

    # 1. Sieve for Smallest Prime Factor (SPF) using NumPy
    spf = np.arange(N + 2, dtype=np.int64)
    sqrt_n = int(math.isqrt(N))
    for i in range(2, sqrt_n + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == i*i, i, spf[i*i::i])

    # 2. Compute Largest Prime Factor (LPF) for all numbers
    # lpf[i] = max(spf[i], lpf[i // spf[i]])
    lpf = np.zeros(N + 2, dtype=np.int64)
    lpf[1] = 1
    for i in range(2, N + 2):
        p = spf[i]
        lpf[i] = max(p, lpf[i // p])

    # 3. Search for n in [10000, N] minimizing max(lpf[n], lpf[n+1])
    # Vectorize the computation of max(lpf[n], lpf[n+1]) for all n in range
    # lpf is 0-indexed, so for n in [10000, N], we need lpf[n] and lpf[n+1]
    # The array of values for n=10000..N is max(lpf[10000:N+1], lpf[10001:N+2])
    n_vals = np.arange(10000, N + 1)
    vals = np.maximum(lpf[n_vals], lpf[n_vals + 1])
    
    # Find the minimum value and its index
    min_idx = np.argmin(vals)
    best_n = int(n_vals[min_idx])
    best_val = int(vals[min_idx])

    return best_n

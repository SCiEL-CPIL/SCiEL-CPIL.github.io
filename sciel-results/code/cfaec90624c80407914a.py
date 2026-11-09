import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s limit

    N_MIN = 10000
    N_MAX = 1000000

    # Direct approach: compute largest prime factor for all numbers in [2, N_MAX+1]
    # using a modified sieve. This is O(N log log N) with small constant.
    max_val = N_MAX + 1

    # lpf[i] = largest prime factor of i
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    lpf[1] = 1

    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime, mark all multiples
            lpf[i::i] = i

    # For each n in [N_MIN, N_MAX], compute max(lpf[n], lpf[n+1])
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    combined = np.maximum(lpf_n, lpf_np1)

    # Find the n with minimum combined LPF
    min_idx = np.argmin(combined)
    best_n = int(n_range[min_idx])

    return best_n

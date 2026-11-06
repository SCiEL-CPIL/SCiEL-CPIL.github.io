import time
import numpy as np
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.perf_counter()
    deadline = start_time + 55.0  # Safety margin

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part
    sp_parts = np.ones(nvals, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        first_idx = (p - lower % p) % p
        idxs = np.arange(first_idx, nvals, p)
        if len(idxs) == 0:
            continue
        
        vals = (lower + idxs).astype(np.int64)
        mult = np.ones(len(idxs), dtype=np.int64)
        
        while True:
            mask = (vals % p == 0)
            if not mask.any():
                break
            vals[mask] //= p
            mult[mask] *= p
        
        sp_parts[idxs] *= mult

    sp_list = sp_parts.tolist()

    best_n = lower
    best_count = float('inf')

    # Initial window
    window_counts = defaultdict(int)
    for i in range(1, 25):
        window_counts[sp_list[i]] += 1
    current_distinct = len(window_counts)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower

    # Slide
    for n in range(lower + 1, upper + 1):
        if time.perf_counter() > deadline:
            break

        out_val = sp_list[n - lower]
        in_val = sp_list[n + 24 - lower]

        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]
        window_counts[in_val] += 1

        current_distinct = len(window_counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count <= 5:
                break

    return best_n

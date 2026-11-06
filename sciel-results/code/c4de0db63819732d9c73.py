"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    # sp_parts[i] = product of all prime powers p^k where p < 24 and p^k divides (lower+i)
    sp_parts = np.ones(nvals, dtype=np.int64)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Find all indices where arr is divisible by p
        mask = (arr % p == 0)
        if not np.any(mask):
            continue
        # For each such index, compute the full p-part
        indices = np.where(mask)[0]
        for idx in indices:
            val = arr[idx]
            while val % p == 0:
                sp_parts[idx] *= p
                val //= p

    # Now search for n in [lower, upper] such that the 24 values
    # sp_parts[n+1-lower], ..., sp_parts[n+24-lower] have minimum distinct count
    # If all are 1, distinct count = 1 (the global minimum)

    # Convert to list for faster access
    sp_list = sp_parts.tolist()

    best_n = lower
    best_count = 999  # max possible distinct count is small

    # Use a frequency dictionary for the sliding window
    window = {}
    
    # Initial window for n=lower: sp_list[1] through sp_list[24]
    for i in range(1, 25):
        v = sp_list[i]
        window[v] = window.get(v, 0) + 1
    distinct = len(window)
    if distinct < best_count:
        best_count = distinct
        best_n = lower
        if best_count == 1:
            return best_n

    # Slide the window
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break
        
        # Remove sp_list[offset] (leaving the window)
        val_out = sp_list[offset]
        window[val_out] -= 1
        if window[val_out] == 0:
            del window[val_out]
        
        # Add sp_list[offset + 24] (entering the window)
        val_in = sp_list[offset + 24]
        window[val_in] = window.get(val_in, 0) + 1
        
        distinct = len(window)
        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
# EVOLVE-BLOCK-END

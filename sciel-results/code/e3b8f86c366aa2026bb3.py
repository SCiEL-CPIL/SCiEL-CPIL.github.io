"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin vs 90s kill

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size

    nvals = max_val - lower + 1

    # ---- Step 1: Compute the 23-smooth part of each integer in [lower, max_val] ----
    # For each number, extract the complete factor composed of primes strictly below 24.
    # We do this by repeatedly dividing by each small prime and accumulating.
    sp_part = np.ones(nvals, dtype=np.int64)
    remaining = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Repeatedly divide by p, accumulating the p-factor in sp_part
        changed = True
        while changed:
            changed = False
            mask = (remaining % p == 0)
            if np.any(mask):
                changed = True
                sp_part[mask] *= p
                remaining[mask] //= p

    if time.time() > deadline:
        return lower

    # ---- Step 2: Map smooth-part values to compact integer IDs ----
    uniq_vals, ids = np.unique(sp_part, return_inverse=True)
    n_unique = len(uniq_vals)
    ids = ids.astype(np.int32)

    if time.time() > deadline:
        return lower

    # ---- Step 3: Sliding window with O(1) frequency updates ----
    # We need windows of size 24: for each n in [lower, upper], consider n+1, ..., n+24
    # That corresponds to indices [1, 24] in our array (0-indexed from lower)
    # So the window starts at index 1 and ends at index 24, then slides.
    # n = lower + i means we look at elements at positions i+1 through i+24 in the array.
    # Position j in the array corresponds to value lower + j.
    # So for n = lower + i, we want elements at positions i+1, i+2, ..., i+24.
    
    max_start = min(nvals - window_size - 1, upper - lower)  # i ranges from 0 to max_start

    freq = np.zeros(n_unique, dtype=np.int32)
    distinct_count = 0

    # Initialize first window: i=0, positions 1 through 24
    for j in range(1, window_size + 1):
        idx = int(ids[j])
        if freq[idx] == 0:
            distinct_count += 1
        freq[idx] += 1

    best_n = lower
    best_count = distinct_count

    # Slide the window
    for i in range(1, max_start + 1):
        # Remove position i (the one leaving the window)
        out_idx = int(ids[i])
        freq[out_idx] -= 1
        if freq[out_idx] == 0:
            distinct_count -= 1
        # Add position i + window_size (the one entering the window)
        in_idx = int(ids[i + window_size])
        if freq[in_idx] == 0:
            distinct_count += 1
        freq[in_idx] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = lower + i
            if best_count <= 2:
                break

        if i % 50000 == 0 and time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END

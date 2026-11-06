"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    sp_parts = np.ones(nvals, dtype=np.int64)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        power = p
        while power <= max_val:
            mask = (arr % power == 0)
            if np.any(mask):
                sp_parts[mask] *= p
            power *= p

    # Check for window with all small-prime parts equal to 1
    # This corresponds to 24 consecutive integers coprime to 23#
    # If found, distinct count is 1.
    is_one = (sp_parts == 1)
    # Use convolution to find sums of windows of size 24
    # We need indices 1..24 for n=lower, so we look at sp_parts[1:25] etc.
    # is_one[1:25] corresponds to n=lower
    # We want to check all n in [lower, upper]
    # Window for n is sp_parts[n+1-lower : n+24-lower+1]
    # Let's create a view of is_one starting from index 1
    # is_one[1 : 1 + (upper - lower + 1) + 24]
    # Actually, simpler:
    # sliding_sum[i] = sum(is_one[i : i+24])
    # We want sliding_sum[i] == 24 for some i corresponding to n+1-lower
    
    # Valid range for start index i:
    # n=lower -> i=1
    # n=upper -> i=upper-lower+1
    # So i in [1, upper-lower+1]
    # Window ends at i+23. Max i+23 = upper-lower+1+23 = upper-lower+24.
    # sp_parts size is max_val - lower + 1 = upper + 24 - lower + 1.
    # Index upper-lower+24 is within bounds.
    
    # We can use np.convolve on the relevant slice
    # Slice is is_one[1 : 1 + (upper - lower + 1) + 24]
    # Length of slice: (upper - lower + 1) + 24
    # Convolve with ones(24) 'valid' mode gives length: (upper - lower + 1)
    # which corresponds to n=lower .. n=upper
    
    if upper >= lower:
        slice_len = (upper - lower + 1) + 24
        if slice_len <= len(is_one):
            window_sums = np.convolve(is_one[1:1+slice_len], np.ones(24, dtype=np.int8), mode='valid')
            # Find if any sum is 24
            winner_indices = np.where(window_sums == 24)[0]
            if len(winner_indices) > 0:
                # First winner
                idx = winner_indices[0]
                # idx corresponds to n = lower + idx
                return lower + idx

    # Fallback to sliding window with dict for general minimum distinct count
    sp_list = sp_parts.tolist()
    
    best_n = lower
    best_count = 999
    
    # Use a frequency dictionary for the sliding window
    window = {}
    
    # Initial window for n=lower: sp_list[1] through sp_list[24]
    for i in range(1, 25):
        v = sp_list[i]
        window[v] = window.get(v, 0) + 1
    distinct = len(window)
    best_count = distinct
    if best_count == 1:
        return best_n

    # Slide the window
    n_steps = upper - lower
    check_interval = 10000
    current_time = start_time
    
    for offset in range(1, n_steps + 1):
        if offset % check_interval == 0:
            current_time = time.time()
            if current_time > deadline:
                break
        
        # Remove sp_list[offset] (leaving the window)
        val_out = sp_list[offset]
        c = window[val_out] - 1
        if c == 0:
            del window[val_out]
        else:
            window[val_out] = c
        
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

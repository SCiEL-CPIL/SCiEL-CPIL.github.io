"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: 90s limit, leave 10s buffer

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size  # 1000024
    
    nvals = max_val - lower + 1  # 1000025
    
    # Precompute smooth parts using vectorized NumPy operations
    smooth_parts = np.ones(nvals, dtype=np.int64)
    nums = np.arange(lower, max_val + 1, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        temp = nums.copy()
        while True:
            mask = (temp % p == 0)
            if not np.any(mask):
                break
            smooth_parts[mask] *= p
            temp[mask] //= p
    
    if time.time() > deadline:
        return lower
    
    # Map to compact IDs
    unique_vals, inverse = np.unique(smooth_parts, return_inverse=True)
    num_ids = len(unique_vals)
    sp_ids = inverse.astype(np.int32)
    
    # Convert to Python list for faster indexing in the sliding window
    sp_list = sp_ids.tolist()
    
    # Sliding window with Python list for freq
    freq = [0] * num_ids
    best_n = lower
    best_count = window_size + 1
    current_distinct = 0
    
    # Initialize window for n=lower: indices 1 to 24
    for i in range(1, window_size + 1):
        val = sp_list[i]
        if freq[val] == 0:
            current_distinct += 1
        freq[val] += 1
    
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower
    
    # Slide window with reduced time-check overhead
    check_every = 50000
    next_check_time = start_time + 0.5
    
    for n in range(lower + 1, upper + 1):
        if (n - lower) % check_every == 0:
            if time.time() > deadline:
                break
            next_check_time += 0.5
        
        out_idx = n - lower
        in_idx = n + 24 - lower
        
        val_out = sp_list[out_idx]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            current_distinct -= 1
        
        val_in = sp_list[in_idx]
        if freq[val_in] == 0:
            current_distinct += 1
        freq[val_in] += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 2:
                break
    
    return best_n
# EVOLVE-BLOCK-END

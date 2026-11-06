"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size
    
    # Step 1: Precompute the "small-prime part" using NumPy for speed
    nvals = max_val - lower + 1
    sp_part = np.ones(nvals, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        # For each power p^k, multiply sp_part by p for all numbers divisible by p^k
        for p in _SMALL_PRIMES:
            power = p
            while power <= max_val:
                first_mult_power = ((lower + power - 1) // power) * power
                if first_mult_power > max_val:
                    break
                idx_array_power = np.arange(first_mult_power, max_val + 1, power) - lower
                sp_part[idx_array_power] *= p
                power *= p
                if power > max_val:
                    break
    if time.time() > deadline:
        return lower

    # Step 2: Slide a window of size 24
    best_n = lower
    best_count = float('inf')
    
    sp_list = sp_part.tolist()
    
    freq = {}
    current_distinct = 0
    
    # Initialize window for n=lower (indices 1 to 24)
    for i in range(1, window_size + 1):
        val = sp_list[i]
        if val not in freq:
            freq[val] = 0
            current_distinct += 1
        freq[val] += 1
        
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower
        
    # Slide window with reduced time-check overhead
    sp_len = len(sp_list)
    n = lower + 1
    upper_check = upper
    deadline_local = deadline
    time_check_interval = 100000
    
    while n <= upper_check:
        if (n - lower) % time_check_interval == 0:
            if time.time() > deadline_local:
                break
        
        out_idx = n - lower
        in_idx = out_idx + 24
        
        val_out = sp_list[out_idx]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
            current_distinct -= 1
            
        val_in = sp_list[in_idx]
        if val_in not in freq:
            freq[val_in] = 0
            current_distinct += 1
        freq[val_in] += 1
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            
        n += 1

    return best_n
# EVOLVE-BLOCK-END

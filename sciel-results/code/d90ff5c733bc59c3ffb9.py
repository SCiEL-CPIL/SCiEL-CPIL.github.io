import time
import numpy as np
from numpy.lib.stride_tricks import as_strided

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    # Using int32 to save memory and potentially speed up operations
    # Max 23-smooth number < 10^6 is well within int32 range
    extracted = np.ones(MAX_VAL + 1, dtype=np.int32)
    
    for p in small_primes:
        # For each power p^k, multiply extracted[m] by p for all m divisible by p^k
        # This correctly accumulates p^(v_p(m)) where v_p(m) is the exponent of p in m
        pk = p
        while pk <= MAX_VAL:
            multiples = np.arange(pk, MAX_VAL + 1, pk)
            extracted[multiples] *= p
            pk *= p

    # We want to find n in [N_MIN, N_MAX] minimizing distinct count in extracted[n+1 : n+1+WINDOW]
    # Total windows to check: N_MAX - N_MIN + 1
    total_windows = N_MAX - N_MIN + 1
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    # Process in chunks to balance memory and speed
    # 100,000 windows * 24 elements * 4 bytes = ~9.6 MB per chunk, which is safe
    CHUNK_SIZE = 200000
    
    # We need extracted values from N_MIN+1 to N_MAX+WINDOW
    # The array segment we care about is extracted[N_MIN+1 : N_MAX+1+WINDOW]
    
    for chunk_start in range(0, total_windows, CHUNK_SIZE):
        if time.time() - start_time > DEADLINE:
            break
        
        chunk_end = min(chunk_start + CHUNK_SIZE, total_windows)
        n_start = N_MIN + chunk_start
        n_end = N_MIN + chunk_end  # exclusive
        
        # Windows start at n_start, n_start+1, ..., n_end-1
        # Each window i (relative to chunk) uses extracted[n_start + i + 1 : n_start + i + 1 + WINDOW]
        # So we need extracted[n_start+1 : n_end + WINDOW]
        arr_start = n_start + 1
        arr_end = n_end + WINDOW
        
        # Create the flat array slice
        flat_arr = extracted[arr_start : arr_end]
        
        # Check if we have enough data (should always be true)
        if flat_arr.size >= chunk_end - chunk_start + WINDOW - 1:
            # Use as_strided for a zero-copy view
            # Shape: (chunk_len, WINDOW)
            # Strides: (element_size, element_size)
            row_stride = flat_arr.strides[0]
            chunk_len = chunk_end - chunk_start
            
            # Create the 2D view
            windows_2d = as_strided(
                flat_arr,
                shape=(chunk_len, WINDOW),
                strides=(row_stride, row_stride)
            )
            
            # Count distinct values per row
            # Sort each row
            sorted_windows = np.sort(windows_2d, axis=1)
            # Number of distinct = 1 + number of positions where value changes
            changes = np.diff(sorted_windows, axis=1)
            distinct_counts = 1 + np.sum(changes != 0, axis=1)
            
            # Find min in this chunk
            local_min_idx = np.argmin(distinct_counts)
            local_min_val = int(distinct_counts[local_min_idx])
            
            if local_min_val < best_count:
                best_count = local_min_val
                best_n = n_start + int(local_min_idx)
                if best_count == 1:
                    # Can't do better than 1
                    break
        else:
            # Fallback: compute directly (shouldn't happen)
            for j in range(chunk_len):
                n = n_start + j
                vals = extracted[n+1 : n+1+WINDOW]
                dc = len(set(vals.tolist()))
                if dc < best_count:
                    best_count = dc
                    best_n = n
                    if best_count == 1:
                        break

    return best_n

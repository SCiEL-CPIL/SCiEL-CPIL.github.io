import time
import numpy as np
from numpy.lib.stride_tricks import as_strided

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Tighter safety margin

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int32)
    
    for p in small_primes:
        pk = p
        while pk <= MAX_VAL:
            multiples = np.arange(pk, MAX_VAL + 1, pk)
            extracted[multiples] *= p
            pk *= p

    total_windows = N_MAX - N_MIN + 1
    
    best_n = N_MIN
    best_count = WINDOW + 1
    
    CHUNK_SIZE = 200000
    
    for chunk_start in range(0, total_windows, CHUNK_SIZE):
        if time.time() - start_time > DEADLINE:
            break
        
        chunk_end = min(chunk_start + CHUNK_SIZE, total_windows)
        n_start = N_MIN + chunk_start
        n_end = N_MIN + chunk_end
        
        arr_start = n_start + 1
        arr_end = n_end + WINDOW
        
        flat_arr = extracted[arr_start : arr_end]
        
        if flat_arr.size >= chunk_end - chunk_start + WINDOW - 1:
            row_stride = flat_arr.strides[0]
            chunk_len = chunk_end - chunk_start
            
            windows_2d = as_strided(
                flat_arr,
                shape=(chunk_len, WINDOW),
                strides=(row_stride, row_stride)
            )
            
            # Sort each row
            sorted_windows = np.sort(windows_2d, axis=1)
            # Count distincts
            changes = np.diff(sorted_windows, axis=1)
            # Use count_nonzero for efficiency
            distinct_counts = 1 + np.count_nonzero(changes != 0, axis=1)
            
            local_min_idx = np.argmin(distinct_counts)
            local_min_val = int(distinct_counts[local_min_idx])
            
            if local_min_val < best_count:
                best_count = local_min_val
                best_n = n_start + int(local_min_idx)
                if best_count == 1:
                    break
        else:
            # Fallback
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

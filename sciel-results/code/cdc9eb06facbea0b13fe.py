import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    # Max value we need: 1000000 + 24 = 1000024
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        # Sieve-style: for each power p^k, multiply extracted by p at each multiple
        power = p
        while power <= MAX_VAL:
            extracted[power::power] *= p
            power *= p

    # Sliding window search using NumPy vectorization
    # For each n in [N_MIN, N_MAX], window = extracted[n+1 : n+WINDOW+1]
    # We want to minimize the number of distinct values in each window.

    # Build the full array of extracted values we need
    # Window for n uses indices n+1 .. n+WINDOW
    # So we need extracted[N_MIN+1 .. N_MAX+WINDOW]
    vals = extracted[N_MIN + 1 : N_MAX + WINDOW + 1]
    # vals[i] corresponds to extracted[N_MIN + 1 + i]
    # Window for n = N_MIN + j uses vals[j : j+WINDOW]

    n_windows = N_MAX - N_MIN + 1
    best_count = WINDOW + 1
    best_n = N_MIN

    # Vectorized distinct count using pairwise first-occurrence detection
    # For each window, distinct = 1 + sum_{j=1}^{W-1} [w[j] not in w[0:j]]
    # Vectorized over chunks using broadcasting for the small WINDOW dimension
    
    CHUNK = 50000
    for chunk_start in range(0, n_windows, CHUNK):
        if time.time() - start_time > DEADLINE:
            break
        chunk_end = min(chunk_start + CHUNK, n_windows)
        n_chunk = chunk_end - chunk_start
        
        chunk_vals = vals[chunk_start : chunk_start + n_chunk + WINDOW - 1]
        
        # Create a contiguous copy of the windows for this chunk
        windows = np.lib.stride_tricks.as_strided(
            chunk_vals,
            shape=(n_chunk, WINDOW),
            strides=(chunk_vals.strides[0], chunk_vals.strides[0])
        ).copy()
        
        # Compute distinct count: start with 1 for the first element,
        # then for each j=1..W-1, add 1 if windows[:,j] doesn't match any earlier
        distinct_counts = np.ones(n_chunk, dtype=np.int32)
        for j in range(1, WINDOW):
            # (n_chunk, 1) vs (n_chunk, j) -> (n_chunk, j)
            matches = (windows[:, j:j+1] == windows[:, :j]).any(axis=1)
            distinct_counts += (~matches).astype(np.int32)
        
        min_idx = int(np.argmin(distinct_counts))
        min_val = int(distinct_counts[min_idx])
        
        if min_val < best_count:
            best_count = min_val
            best_n = N_MIN + chunk_start + min_idx
            if best_count <= 2:
                return best_n

    return best_n

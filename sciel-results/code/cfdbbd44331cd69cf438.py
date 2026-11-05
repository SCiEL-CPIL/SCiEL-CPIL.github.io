import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m]: the part of m composed of primes < 24
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        temp = multiples.copy()
        p_parts = np.ones(len(multiples), dtype=np.int64)
        # Extract all powers of p from each multiple
        while np.any(temp % p == 0):
            mask = temp % p == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts

    # Map extracted values to compact integer IDs
    # This allows using a numpy array for frequency counting instead of a dict
    unique_vals = np.unique(extracted[1:])
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    n_ids = len(unique_vals)

    # Convert extracted to ID array
    extracted_ids = np.zeros(MAX_VAL + 1, dtype=np.int32)
    for i in range(1, MAX_VAL + 1):
        extracted_ids[i] = val_to_id[extracted[i]]

    # Sliding window with numpy frequency array
    freq = np.zeros(n_ids, dtype=np.int32)

    # Initialize window for n = N_MIN: covers [N_MIN+1, N_MIN+24]
    window_start = N_MIN + 1
    window_end = N_MIN + WINDOW  # inclusive
    for idx in range(window_start, window_end + 1):
        freq[extracted_ids[idx]] += 1

    distinct_count = np.count_nonzero(freq)
    best_n = N_MIN
    best_count = distinct_count

    # Slide window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break

        # Remove extracted[n] (was at position n in the previous window)
        out_id = extracted_ids[n]
        freq[out_id] -= 1
        if freq[out_id] == 0:
            distinct_count -= 1

        # Add extracted[n + WINDOW]
        in_id = extracted_ids[n + WINDOW]
        if freq[in_id] == 0:
            distinct_count += 1
        freq[in_id] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break

    return best_n

import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts

    # Map extracted values to compact integer indices for fast array-based counting
    # extracted[1..MAX_VAL] -> inverse[0..MAX_VAL-1]
    unique_vals, inverse = np.unique(extracted[1:], return_inverse=True)
    inverse = inverse.astype(np.int32)

    n_unique = len(unique_vals)
    count_arr = np.zeros(n_unique, dtype=np.int32)

    # Window for n covers extracted[n+1 .. n+24] -> inverse[n .. n+23]
    # Initialize for n = N_MIN
    window_slice = inverse[N_MIN : N_MIN + WINDOW]
    np.add.at(count_arr, window_slice, 1)
    distinct_count = int(np.count_nonzero(count_arr))

    best_n = N_MIN
    best_count = distinct_count

    # Slide window: from n to n+1, remove inverse[n], add inverse[n+23]
    # Check time periodically to avoid per-iteration overhead
    CHECK_INTERVAL = 10000
    for n in range(N_MIN + 1, N_MAX + 1):
        if (n - N_MIN) % CHECK_INTERVAL == 0:
            if time.time() - start_time > DEADLINE:
                break

        out_idx = inverse[n - 1]
        count_arr[out_idx] -= 1
        if count_arr[out_idx] == 0:
            distinct_count -= 1

        in_idx = inverse[n + WINDOW - 1]
        if count_arr[in_idx] == 0:
            distinct_count += 1
        count_arr[in_idx] += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break

    return best_n

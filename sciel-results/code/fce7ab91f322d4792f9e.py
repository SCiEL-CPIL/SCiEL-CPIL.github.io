"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    N = max_val - lower + 1

    # Compute smooth part (factor composed of primes < 24) for each number
    # in [lower, max_val]
    # smooth[i] = the part of (lower + i) made of primes < 24
    # Method: for each number, divide out all prime factors >= 24
    # Equivalently: multiply together all p^k for p in {2,3,5,7,11,13,17,19,23}
    
    smooth = np.arange(lower, max_val + 1, dtype=np.int64)
    
    # For each small prime p, extract the p-adic part
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    result = np.ones(N, dtype=np.int64)
    
    for p in small_primes:
        # Extract p-power from each number
        temp = smooth.copy()
        factor = np.ones(N, dtype=np.int64)
        while True:
            mask = (temp % p) == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            factor[mask] *= p
        result *= factor
    
    smooth = result
    # Now smooth[i] is the smooth part of (lower + i)
    # Sliding window: for each n in [lower, upper], look at smooth[n+1-lower] to smooth[n+24-lower]
    # Count distinct values in that window

    # Use a dictionary for frequency counting
    best_n = lower
    best_count = 25

    freq = {}
    distinct = 0

    # Initialize window for n = lower: indices 1 to 24 in smooth array
    for i in range(1, 25):
        v = int(smooth[i])
        if v not in freq:
            freq[v] = 0
            distinct += 1
        freq[v] += 1

    if distinct < best_count:
        best_count = distinct
        best_n = lower

    # Slide the window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        out_idx = n - lower
        in_idx = n + 24 - lower

        v_out = int(smooth[out_idx])
        freq[v_out] -= 1
        if freq[v_out] == 0:
            del freq[v_out]
            distinct -= 1

        v_in = int(smooth[in_idx])
        if v_in not in freq:
            freq[v_in] = 0
            distinct += 1
        freq[v_in] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n

    return best_n
# EVOLVE-BLOCK-END

"""Rewritten program: systematic search minimizing distinct small-prime factors."""
import time
import math
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin under 60s

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    P = len(primes)

    import numpy as np

    # Range of values: from 1001 to 1000000
    lo = 1001
    hi = 1000000
    size = hi - lo + 1

    # Precompute small_prime_part for all values in [lo, hi] using vectorized NumPy
    # small_prime_part(x) = product of p^(v_p(x)) for p in primes
    # For each prime p, repeatedly divide out p and multiply kept by p
    kept = np.ones(size, dtype=np.int64)
    rem = np.arange(lo, hi + 1, dtype=np.int64)

    for p in primes:
        # Repeatedly extract factor p from rem, multiplying kept by p each time
        while True:
            mask = rem % p == 0
            if not mask.any():
                break
            rem[mask] //= p
            kept[mask] *= p

    # Convert to list for faster indexing in the sliding window
    kept_list = kept.tolist()

    # For each n in [1000, 1000000-24], the 24 values are kept_list[n-1000 : n-977]
    # (indices 0-based: n+i for i=1..24 maps to array index (n+i)-lo = n+i-1001)
    # So for n, indices are n-1000 through n-977 inclusive (24 values)

    n_max = 1000000 - 24

    # Sliding window with frequency tracking
    # Use a dict for frequency counts
    freq = {}
    distinct = 0

    # Initialize window for n=1000: indices 0 to 23
    for i in range(24):
        v = kept_list[i]
        if v not in freq:
            freq[v] = 0
            distinct += 1
        freq[v] += 1

    best_count = distinct
    best_n = 1000

    # Slide the window
    for n in range(1001, n_max + 1):
        # Remove kept_list[n-1001] (leaving the window)
        out_val = kept_list[n - 1001]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
            distinct -= 1

        # Add kept_list[n-977] (entering the window)
        in_val = kept_list[n - 977]
        if in_val not in freq:
            freq[in_val] = 0
            distinct += 1
        freq[in_val] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break

        if time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END

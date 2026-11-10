import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024

    # Precompute small prime part using vectorized sieve-like method
    small_part = np.ones(max_val + 1, dtype=np.int64)

    for p in primes:
        p_power = p
        while p_power <= max_val:
            indices = np.arange(p_power, max_val + 1, p_power)
            small_part[indices] *= p
            p_power *= p

    # Convert to Python list for fast dictionary lookups in sliding window
    sp = small_part.tolist()

    # Sliding window frequency count
    WINDOW = 24
    best_count = float('inf')
    best_n = 1000

    # Initialize window for n=1000: numbers 1001 to 1024
    freq = defaultdict(int)
    distinct = 0
    for i in range(1001, 1001 + WINDOW):
        val = sp[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1

    best_count = distinct
    best_n = 1000

    # Slide window
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break

        # Remove sp[n] (which was n-1+1 = n for previous window)
        old_val = sp[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct -= 1

        # Add sp[n + WINDOW]
        new_val = sp[n + WINDOW]
        if freq[new_val] == 0:
            distinct += 1
        freq[new_val] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break

    return best_n

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

    # Now we need to find n in [N_MIN, N_MAX] that minimizes the number of
    # distinct values in extracted[n+1 : n+WINDOW+1]
    #
    # Strategy: Use a sliding window with a frequency dictionary.
    # For each n, we maintain a count of each extracted value in the window.
    # As we slide from n to n+1, we remove extracted[n+1] and add extracted[n+WINDOW+1].
    # We track the number of distinct values (those with count > 0).

    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct

    # We'll iterate n from N_MIN to N_MAX
    # Window for n: extracted[n+1], extracted[n+2], ..., extracted[n+WINDOW]
    # i.e., indices n+1 through n+WINDOW

    # Initialize the window for n = N_MIN
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0

    for i in range(1, WINDOW + 1):
        val = extracted[N_MIN + i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1

    if distinct < best_count:
        best_count = distinct
        best_n = N_MIN
        if best_count <= 2:
            return best_n

    # Slide the window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break

        # Remove extracted[n] (which was at position n, i.e., index n in the array)
        # Wait, let me re-index. Window for n is extracted[n+1] ... extracted[n+WINDOW]
        # Window for n-1 is extracted[n] ... extracted[n+WINDOW-1]
        # So when going from n-1 to n:
        #   Remove extracted[n] (the leftmost of the previous window)
        #   Add extracted[n+WINDOW] (the new rightmost)

        # Remove extracted[n]
        val_out = extracted[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct -= 1

        # Add extracted[n + WINDOW]
        val_in = extracted[n + WINDOW]
        if freq[val_in] == 0:
            distinct += 1
        freq[val_in] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 2:
                return best_n

    return best_n

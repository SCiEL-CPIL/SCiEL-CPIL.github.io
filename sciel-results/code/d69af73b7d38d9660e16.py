import numpy as np
import time

def construct():
    start_time = time.time()
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    DEADLINE = start_time + 50.0

    # 1. Sieve of Eratosthenes to find primes up to LIMIT
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False

    for i in range(2, int(np.sqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i : LIMIT+1 : i] = False

    primes = np.where(is_prime)[0]

    # 2. Compute Largest Prime Factor (LPF) for all numbers up to LIMIT
    lpf = np.arange(LIMIT + 1, dtype=np.int64)

    for p in primes:
        if time.time() > DEADLINE:
            break
        lpf[p : LIMIT+1 : p] = p

    # 3. Find n in [10000, MAX_N] minimizing max(lpf[n], lpf[n+1])
    n_start = 10000
    n_end = MAX_N

    vals_n = lpf[n_start : n_end + 1]
    vals_np1 = lpf[n_start + 1 : n_end + 2]

    max_lpf = np.maximum(vals_n, vals_np1)

    min_val = np.min(max_lpf)
    best_idx = np.argmin(max_lpf)
    best_n = n_start + best_idx

    # 4. If time remains, try to find a better solution using a different approach
    # This is a safeguard; the global minimum is likely already found.
    remaining_time = DEADLINE - time.time()
    if remaining_time > 10.0:
        # Try a local search around the best candidate to see if there's a tie or better
        # But since we did a full scan, this is redundant. Instead, we can verify
        # that the result is indeed minimal by checking a few random points.
        # However, the full scan is O(N) and already done, so no better solution exists.
        pass

    return int(best_n)

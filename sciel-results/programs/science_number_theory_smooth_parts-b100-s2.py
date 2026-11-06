"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    sp_parts = [1] * nvals

    for p in _SMALL_PRIMES:
        first_mult = ((lower + p - 1) // p) * p
        for m in range(first_mult, max_val + 1, p):
            idx = m - lower
            temp = m
            while temp % p == 0:
                sp_parts[idx] *= p
                temp //= p

    # Scan for best n: minimize number of distinct small-prime parts in [n+1, n+24]
    best_n = lower
    best_count = float('inf')

    for n in range(lower, upper + 1):
        if time.time() > deadline:
            break
        start_idx = (n + 1) - lower
        end_idx = (n + 24) - lower
        seen = set()
        for i in range(start_idx, end_idx + 1):
            seen.add(sp_parts[i])
        cnt = len(seen)
        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 2:
                pass  # keep scanning in case we find even lower

    return best_n
# EVOLVE-BLOCK-END

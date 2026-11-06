import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    # sp[i] = largest divisor of (lower+i) composed only of primes in _SMALL_PRIMES
    sp = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # For each multiple of p, divide out all factors of p
        mask = (sp % p == 0)
        # We need to divide out ALL factors of p, so iterate
        while np.any(mask):
            sp[mask] //= p
            mask = (sp[mask] % p == 0) if np.any(mask) else np.zeros(0, dtype=bool)
            # Rebuild mask: indices where current sp is still divisible by p
            if np.any(mask):
                mask = (sp % p == 0)
            else:
                break

    # Actually the above while loop is tricky with numpy. Let me use a different approach:
    # For each number, repeatedly divide by each small prime.
    # Better: for each small prime p, for each multiple of p, divide out all p factors.
    # Let me redo this more carefully.

    sp = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # Find all indices where sp is divisible by p
        # We need to divide out ALL factors of p
        # Use a loop: while any element is divisible by p, divide those
        while True:
            mask = (sp % p == 0)
            if not np.any(mask):
                break
            sp[mask] //= p

    # Now sp[i] is the small-prime part of (lower + i)
    # Convert to list for faster dictionary operations
    sp_list = sp.tolist()

    # Sliding window over 24 consecutive values
    # Window for n: indices [n-lower, n-lower+23] in sp_list, corresponding to n+1 ... n+24
    # Wait: n+i for i=1..24 means values n+1, n+2, ..., n+24
    # In sp_list, index for value v is v - lower
    # So n+1 -> index (n+1) - lower, n+24 -> index (n+24) - lower

    best_n = lower
    best_count = float('inf')

    # Initialize window for n = lower: values lower+1 to lower+24
    # indices: (lower+1)-lower = 1, ..., (lower+24)-lower = 24
    from collections import Counter

    # Window indices: for n, the values are at sp_list[n+1-lower] through sp_list[n+24-lower]
    # So offset = n - lower, window is sp_list[offset+1 : offset+25]

    # Let's just iterate
    for n in range(lower, upper + 1):
        if time.time() > deadline:
            break

        # Window: sp_list[n+1-lower] to sp_list[n+24-lower]
        start_idx = n + 1 - lower
        end_idx = n + 24 - lower  # inclusive

        # Use a Counter for the 24 values
        # To avoid creating a new Counter each time, use sliding window
        # But for simplicity and correctness, let's use a sliding approach

        # Actually, let me just use a set for small windows - 24 elements
        # But that's O(24) per step. For 1M steps, that's 24M operations. Should be OK.

        seen = set()
        for i in range(start_idx, end_idx + 1):
            seen.add(sp_list[i])
        cnt = len(seen)

        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 1:
                break  # Can't do better than 1

    return best_n

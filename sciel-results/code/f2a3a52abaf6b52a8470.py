import time
import numpy as np
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 90s kill

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    # sp[i] = largest divisor of (lower+i) composed only of primes in _SMALL_PRIMES
    sp = np.arange(lower, max_val + 1, dtype=np.int64)

    for p in _SMALL_PRIMES:
        # For each multiple of p, divide out all factors of p
        # We need to keep only the p-part, so we divide out everything else
        # Actually, we want the part composed of primes < 24, so we keep p factors
        # and remove all other factors.
        # Better approach: for each number, compute the small-prime part directly.
        # But that's slow. Let's use the sieve approach correctly.
        # 
        # The small-prime part of x is the product of p^a for p in _SMALL_PRIMES
        # where a = v_p(x).
        #
        # We can compute this by: result = 1; for each p in _SMALL_PRIMES:
        #   while x % p == 0: x //= p; result *= p
        #
        # To vectorize: for each p, we need to multiply result by p^(v_p(x)) for all x.
        # We can do this by: temp = x; count = 0; while temp % p == 0: temp //= p; count += 1
        # Then result *= p**count.
        #
        # But doing this in a loop over the array is slow. Let's try a different approach.
        #
        # Actually, the most efficient way is to use the fact that we can divide out
        # all factors of p from x, and the quotient is the part with p removed.
        # But we want to keep the p-part, not remove it.
        #
        # Let me think differently. The small-prime part is the largest divisor of x
        # that is composed only of primes < 24. This is equivalent to x divided by
        # the largest divisor of x that is coprime to the product of primes < 24.
        #
        # But that's not helpful. Let's just compute it directly with a loop, but
        # optimize by only iterating over multiples of each prime.
        #
        # For each prime p, for each multiple of p, we can compute p^(v_p(x)) and
        # multiply it into the result.
        #
        # Let's try this:
        sp_parts = np.ones(nvals, dtype=np.int64)
        for p in _SMALL_PRIMES:
            # For each multiple of p in [lower, max_val]
            first_mult = ((lower + p - 1) // p) * p
            for m in range(first_mult, max_val + 1, p):
                idx = m - lower
                temp = m
                while temp % p == 0:
                    sp_parts[idx] *= p
                    temp //= p

    # Now sp_parts[i] is the small-prime part of (lower + i)
    # Convert to list for faster dictionary operations
    sp_list = sp_parts.tolist()

    # Sliding window over 24 consecutive values
    # Window for n: values n+1, n+2, ..., n+24
    # In sp_list, index for value v is v - lower
    # So n+1 -> index (n+1) - lower, n+24 -> index (n+24) - lower

    best_n = lower
    best_count = float('inf')

    # Initialize window for n = lower: values lower+1 to lower+24
    # indices: (lower+1)-lower = 1, ..., (lower+24)-lower = 24
    window_counts = defaultdict(int)
    for i in range(1, 25):
        window_counts[sp_list[i]] += 1
    current_distinct = len(window_counts)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower

    # Slide the window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        # Remove the value that's leaving: n (index n - lower)
        # Add the value that's entering: n+24 (index n+24 - lower)
        out_val = sp_list[n - lower]
        in_val = sp_list[n + 24 - lower]

        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]
        window_counts[in_val] += 1

        current_distinct = len(window_counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count <= 1:
                break  # Can't do better than 1

    return best_n

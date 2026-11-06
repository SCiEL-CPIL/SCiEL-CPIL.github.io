"""Rewritten program: systematic search minimizing distinct small-prime factors."""
import time
import math
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 60s

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    P = len(primes)

    def small_prime_part(x):
        """Return the factor of x composed of primes < 24 (i.e., remove primes >= 24)."""
        for p in primes:
            if x % p == 0:
                while x % p == 0:
                    x //= p
        return x

    best = 1000
    best_count = None

    # Systematic search over all n in [1000, 1000000]
    # For each n, compute small_prime_part(n+i) for i in 1..24
    # Count distinct values; minimize that count
    n = 1000
    N_max = 1000000
    # We need n+i <= N_max for i up to 24, so n <= N_max - 24
    n_max = N_max - 24

    # Precompute small_prime_part for the full range we need: [1001, 1000000]
    # That's about 999000 values. Compute them once.
    # Use a sieve-like approach for speed: for each prime p, divide out all multiples.
    # Actually, small_prime_part(x) = x with all prime factors >= 24 removed.
    # Equivalently, it's the largest divisor of x that is coprime to the product of primes >= 24... 
    # No wait: it's the part made of primes < 24. So it's the product of p^a for p in primes.
    # That's x / (part with primes >= 24). 
    # Let's just compute it directly but vectorize with NumPy for speed.

    import numpy as np

    # Range of values: from 1001 to 1000000
    lo = 1001
    hi = 1000000
    size = hi - lo + 1
    vals = np.arange(lo, hi + 1, dtype=np.int64)

    # For each prime p < 24, we want to keep p^a in the factor.
    # small_prime_part(x) = product over p in primes of p^(v_p(x))
    # We can compute this by: for each p, compute p^v_p(x) for all x, then multiply.
    # p^v_p(x) = x / (x / p^v_p(x)) ... circular. 
    # Better: repeatedly divide by p and count, but we need the kept part.
    # Alternative: for each x, spf(x) = x; for each prime q >= 24, divide out q.
    # But there are many primes >= 24. 
    # The kept part is the product of p^a for p in {2,3,5,7,11,13,17,19,23}.
    # Let's compute it by: result = 1; for each p in primes: while x%p==0: x//=p; result*=p
    # We can do this with NumPy by iterating primes and using integer division.

    # Actually, the most efficient: precompute an array `kept` initialized to 1.
    # For each prime p in primes:
    #   For each x where x is divisible by p, we need to multiply kept by p for each power.
    # This is complex to vectorize cleanly. 

    # Simpler: just compute small_prime_part for each x in a loop, but optimize with early exit.
    # 999000 values * 9 primes = ~9M operations, which is fast in Python if we're careful.
    # Let's try a pure Python loop first and see if it fits in time.

    kept = [1] * size
    for p in primes:
        for i in range(size):
            v = vals[i]
            if v % p == 0:
                while v % p == 0:
                    v //= p
                    kept[i] *= p
            # Note: we're modifying a local copy v, but we need the original for next primes.
            # This approach is wrong because we modify v. Let me fix.

    # Let me redo: for each x, compute kept part.
    kept = np.ones(size, dtype=np.int64)
    for p in primes:
        # For each x, multiply kept by p^(v_p(x))
        # We can compute p^(v_p(x)) by: temp = x; count=0; while temp%p==0: temp//=p; count+=1
        # Then kept *= p**count
        # Vectorize: 
        temp = vals.copy()
        # We need to do this for each x individually since division is data-dependent.
        # Use a loop over the array for each prime.
        for i in range(size):
            t = temp[i]
            if t % p == 0:
                mult = 1
                while t % p == 0:
                    t //= p
                    mult *= p
                kept[i] *= mult

    # Now kept[i] = small_prime_part(lo + i)
    # For each n in [1000, 1000000-24], the 24 values are kept[n-1001+1 : n-1001+25] = kept[n-lo+1 : n-lo+25]
    # n-lo+1 = n - 1000, n-lo+25 = n - 976
    # So indices from n-1000 to n-977 (inclusive), which is 24 values.

    # Let's search for the minimum distinct count
    best_count = 25  # max possible
    best_n = 1000

    # We can early-stop if we find count = 1 (all same) or count = 2, etc.
    # But let's just scan all and find the min.
    # To speed up, we can use a sliding window and a Counter, but updating a Counter for each shift
    # is O(1) amortized if we track counts.

    from collections import defaultdict

    # Initial window: n=1000, indices 0 to 23
    window_counts = defaultdict(int)
    for i in range(24):
        window_counts[kept[i]] += 1
    current_distinct = len(window_counts)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000

    # Slide the window
    for n in range(1001, n_max + 1):
        # Remove kept[n-1001] (the one leaving), add kept[n-1001+24] = kept[n-977]
        out_val = kept[n - 1001]
        in_val = kept[n - 977]
        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]
        window_counts[in_val] += 1
        current_distinct = len(window_counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count <= 2:  # Can't get much better than 2 (or 1)
                # We can break early if we find a very low count
                # But let's continue to see if we find 1
                if best_count == 1:
                    break
        if time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END

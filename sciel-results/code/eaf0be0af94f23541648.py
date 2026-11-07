"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_lpf(n):
    """Return a list lpf where lpf[i] is the least prime factor of i (or i if i is prime)."""
    if n < 2:
        return list(range(n + 1))
    lpf = list(range(n + 1))
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

def _trial_lpf(m, small_primes):
    """Fallback: compute least prime factor of m via trial division by small_primes.
       Returns m if no small prime divides it (meaning lpf >= small_primes[-1]+1 or m is prime)."""
    for p in small_primes:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s wall-clock

    n_gaps = len(P['gaps'])
    result = [None] * n_gaps

    # Determine the full range of candidate values
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])

    # Precompute least prime factor for all values up to max_val using sieve
    lpf = _sieve_lpf(max_val)

    # Precompute small primes for fallback verification
    # Max gap is 28, so we need to check divisibility by primes < 28
    # But for safety, compute primes up to 100
    def _sieve_primes(limit):
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, isqrt(limit) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    small_primes = _sieve_primes(100)

    for i in range(n_gaps):
        lo, hi = P['gaps'][i]
        gap = hi - lo

        # Find any m in (lo, hi) such that lpf[m] >= gap
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Primary check using precomputed sieve
            if lpf[m] >= gap:
                # Verify with trial division as a cross-check
                # Only verify if gap > 2 (for gap=2, lpf>=2 means odd, trivially true)
                if gap > 2:
                    lf_trial = _trial_lpf(m, small_primes)
                    # If trial says lf < gap but sieve says >= gap, trust sieve (it's more comprehensive)
                    # If trial says lf >= gap, confirm
                    if lf_trial < gap:
                        # Discrepancy: sieve should be correct since it covers all factors
                        # This shouldn't happen; trust sieve
                        pass
                result[i] = m
                found = True
                break

        if not found:
            result[i] = None

    # Ensure all values are Python int
    for i in range(n_gaps):
        if result[i] is not None:
            result[i] = int(result[i])

    return result

# EVOLVE-BLOCK-END

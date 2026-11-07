"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n (inclusive)."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, math.isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _least_prime_factor(m, small_primes):
    """Return the least prime factor of m, using small_primes for trial division.
    If no small prime divides m, check if m itself is prime (m is the LPF).
    Otherwise, do a full trial division up to isqrt(m)."""
    for p in small_primes:
        if p * p > m:
            break
        if m % p == 0:
            return p
    # No small prime factor found. Check if m is prime.
    # If m is prime, LPF = m.
    # Otherwise, find the least factor.
    # Since m < ~10300, this is fast.
    limit = math.isqrt(m)
    for d in range(small_primes[-1] + 1, limit + 1):
        if m % d == 0:
            return d
    return m  # m is prime

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin within 60s limit
    
    # Precompute primes up to max possible gap + some margin
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes up to max_gap to check LPF >= gap
    # Also need primes up to max value for trial division if needed
    max_val = max(hi for lo, hi in P['gaps'])
    primes = _primes_up_to(max_val)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find the first m in (lo, hi) with LPF(m) >= gap
        for m in range(lo + 1, hi):
            lpf = _least_prime_factor(m, primes)
            if lpf >= gap:
                result[i] = m
                break
    
    # Ensure types are correct
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END

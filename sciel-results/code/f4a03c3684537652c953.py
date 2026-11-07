"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return list of primes <= n as Python ints."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s wall-clock
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes = _sieve(max_gap)  # primes up to max gap, as Python ints
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # primes strictly less than gap
        small_primes = [p for p in primes if p < gap]
        for m in range(lo + 1, hi):
            # m's least prime factor >= gap  <=>  m not divisible by any prime < gap
            ok = True
            for p in small_primes:
                if m % p == 0:
                    ok = False
                    break
            if ok:
                result[i] = m  # m is a Python int from range
                break
    return result
# EVOLVE-BLOCK-END

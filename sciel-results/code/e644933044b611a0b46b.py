"""Rewritten program: find witnesses in every consecutive-prime gap."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061],
              [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091],
              [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111],
              [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151],
              [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177],
              [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223],
              [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259],
              [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Sieve of Eratosthenes: all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(sieve[start:n + 1:step]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start = time.monotonic()
    deadline = start + 85.0  # safety margin before the 90 s kill

    # Max gap among the pinned pairs -> least prime factor must be >= gap.
    # A candidate's least factor < gap is impossible unless it is divisible
    # by a prime < gap, so testing primes < gap is sufficient.
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # Sieve a little beyond to be safe (primes strictly below max_gap are what we need).
    small_primes = _primes_up_to(max(30, max_gap))  # primes up to >= max_gap

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.monotonic() > deadline:
            break
        gap = hi - lo
        # Only primes strictly less than gap can disqualify a candidate.
        check_primes = [p for p in small_primes if p < gap]
        found = None
        for m in range(lo + 1, hi):
            ok = True
            for p in check_primes:
                if m % p == 0:
                    ok = False
                    break
            if ok:
                found = m
                break
        result[i] = found

    # Ensure exact types: Python ints (not numpy/bool), list container.
    cleaned = []
    for v in result:
        if v is None:
            cleaned.append(None)
        else:
            cleaned.append(int(v))
    return cleaned
# EVOLVE-BLOCK-END

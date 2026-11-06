"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
MAX_PRIME = 23

def _extract_factor(x):
    """Extract the part of x composed of primes strictly below 24."""
    res = 1
    for p in PRIMES:
        while x % p == 0:
            res *= p
            x //= p
    return res

def _score(n):
    """Number of distinct extracted factors for n+1..n+24."""
    s = set()
    for i in range(1, 25):
        s.add(_extract_factor(n + i))
    return len(s)

def _search_window(lo, hi, deadline):
    """Scan [lo,hi) for n with minimal distinct-factor count."""
    best_n = None
    best_score = 10**9
    # Fast pre-filter: skip n where any n+i has a prime factor >= 24 that is large
    # (not a strong filter, but we just scan and compute exactly)
    for n in range(lo, hi):
        if time.time() >= deadline:
            break
        sc = _score(n)
        if sc < best_score:
            best_score = sc
            best_n = n
            if best_score == 1:
                return best_n, best_score
    return best_n, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s kill

    best_n = 1000
    best_score = _score(1000)

    # First pass: coarse scan with step 1 over a moderate range to find a good seed
    # Use a larger stride initially to cover more ground, then refine.
    # We'll do a multi-resolution scan.
    ranges = [
        (1000, 20000),
        (20000, 100000),
        (100000, 500000),
        (500000, 1000000),
    ]

    # Phase 1: quick scan with step 10 to find promising regions
    for lo, hi in ranges:
        if time.time() >= deadline:
            break
        step = 10
        for n in range(lo, hi, step):
            if time.time() >= deadline:
                break
            sc = _score(n)
            if sc < best_score:
                best_score = sc
                best_n = n
                if best_score == 1:
                    return best_n

    # Phase 2: refine around best found with step 1
    if best_n is not None and best_score > 1:
        lo = max(1000, best_n - 500)
        hi = min(1000000, best_n + 500)
        if time.time() < deadline:
            n2, s2 = _search_window(lo, hi, deadline)
            if n2 is not None and s2 < best_score:
                best_n = n2
                best_score = s2

    # Phase 3: if still not 1, do a denser random/sample scan
    if best_score > 1 and time.time() < deadline:
        import random
        rng = random.Random(42)
        for _ in range(200000):
            if time.time() >= deadline:
                break
            n = rng.randint(1000, 999976)
            sc = _score(n)
            if sc < best_score:
                best_score = sc
                best_n = n
                if best_score == 1:
                    break

    return best_n
# EVOLVE-BLOCK-END

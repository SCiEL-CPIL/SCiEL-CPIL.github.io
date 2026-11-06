"""
Find prime p (<= 10^9) and max k (<= 200) such that p^2+1, ..., p^2+k
all have prime factors <= p.

Strategy:
  1. Precompute primes up to 10^6.
  2. For each candidate prime p, use a segmented-sieve-like trial division
     over the 200 numbers p^2+1..p^2+200 to strip out all small prime factors.
  3. If any remaining factor exceeds p, that number is not p-smooth.
  4. Only call Pollard-Brent when a remaining composite factor is > p and
     we need to determine whether it has a prime factor > p.
  5. Search primes in increasing order; stop early if k=200 is found.
"""

import math
import random
import time

# ---------------------------------------------------------------------------
# Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
# ---------------------------------------------------------------------------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == sp:
            return True
        if n % sp == 0:
            return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------------------------------------------------------------------------
# Pollard-Brent
# ---------------------------------------------------------------------------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

# ---------------------------------------------------------------------------
# Precompute primes up to SIEVE_LIMIT via simple sieve
# ---------------------------------------------------------------------------
SIEVE_LIMIT = 10**6
_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
_sieve[0] = _sieve[1] = 0
for _i in range(2, int(SIEVE_LIMIT ** 0.5) + 1):
    if _sieve[_i]:
        _sieve[_i * _i :: _i] = bytearray(len(_sieve[_i * _i :: _i]))
PRIMES = [i for i in range(2, SIEVE_LIMIT + 1) if _sieve[i]]

# ---------------------------------------------------------------------------
# Check whether all prime factors of v are <= p (p-smooth)
# ---------------------------------------------------------------------------
def has_factor_above(v, p):
    """Return True if v has a prime factor > p."""
    if v <= 1:
        return False
    if v <= p:
        return False
    # Trial division by precomputed primes up to min(p, sqrt(v))
    remaining = v
    for sp in PRIMES:
        if sp > p:
            break
        if sp * sp > remaining:
            break
        while remaining % sp == 0:
            remaining //= sp
    if remaining == 1:
        return False
    if remaining <= p:
        return False
    # remaining > p: if prime, definitely has factor > p
    if is_prime_mr(remaining):
        return True
    # remaining is composite and > p; use Pollard-Brent recursively
    def _hf(m):
        if m <= 1:
            return False
        if m <= p:
            return False
        if is_prime_mr(m):
            return m > p
        d = pollard_brent(m)
        return _hf(d) or _hf(m // d)
    return _hf(remaining)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    best_k = 0
    best_p = 2
    t0 = time.time()
    TIME_LIMIT = 110  # leave 10 s margin

    # --- Phase 1: exhaustive search over primes up to 200 000 ---
    for p in PRIMES:
        if p > 200_000:
            break
        if time.time() - t0 > TIME_LIMIT:
            break
        k = 0
        for i in range(1, 201):
            v = p * p + i
            if has_factor_above(v, p):
                break
            k = i
        if k > best_k:
            best_k = k
            best_p = p
        if k == 200:
            break

    # --- Phase 2: sampled search over primes up to 10^7 ---
    if time.time() - t0 < TIME_LIMIT:
        # Use a stride to sample; for larger p the chance of long runs drops
        # but we still want to probe.
        step = 2
        for p in range(200_001, 10_000_000, step):
            if time.time() - t0 > TIME_LIMIT:
                break
            if not is_prime_mr(p):
                continue
            k = 0
            for i in range(1, 201):
                v = p * p + i
                if has_factor_above(v, p):
                    break
                k = i
            if k > best_k:
                best_k = k
                best_p = p
            if k == 200:
                break

    # --- Phase 3: sparse search up to 10^8 if time remains ---
    if time.time() - t0 < TIME_LIMIT:
        for p in range(10_000_001, 100_000_000, 10_000):
            if time.time() - t0 > TIME_LIMIT:
                break
            if not is_prime_mr(p):
                continue
            k = 0
            for i in range(1, 201):
                v = p * p + i
                if has_factor_above(v, p):
                    break
                k = i
            if k > best_k:
                best_k = k
                best_p = p
            if k == 200:
                break

    return {'p': best_p, 'k': best_k}

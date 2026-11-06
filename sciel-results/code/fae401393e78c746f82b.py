import time
import random
import math

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

# Extend small primes for faster stripping
def _gen_primes_upto(limit):
    comp = bytearray(limit + 1)
    for i in range(2, int(math.isqrt(limit)) + 1):
        if not comp[i]:
            comp[i*i::i] = b'\x01' * ((limit - i*i)//i + 1)
    return [i for i in range(2, limit+1) if not comp[i]]

_STRIPE_PRIMES = _gen_primes_upto(2000)

def is_prime(n):
    """Deterministic Miller-Rabin.  Exact for n < 2^64 with 12 base primes."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ── Pollard's Rho / Brent factorisation ──────────────────────────────────────

def _pollard_brent(n):
    """Return a non-trivial factor of composite n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    c = random.randrange(1, n - 1)
    y = random.randrange(0, n - 1)
    m = 128
    g = r = q = 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        g = 1
        while g == 1:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
    if g == n:
        return _pollard_brent(n)
    return g

def all_prime_factors_leq(n, limit):
    """True iff every prime factor of n is ≤ limit.  Early-exit on violation."""
    if n <= 1:
        return True
    # quick small-prime scan
    for p in _STRIPE_PRIMES:
        if p > limit:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
    if n == 1:
        return True
    # remaining n has no small-prime factors (≤ 2000)
    if n <= limit:
        return True
    if n > limit and is_prime(n):
        return False
    # composite and > limit: need to check if it has any prime factor > limit
    # Try to factor; if any prime factor > limit, return False
    def _has_large_factor(m):
        """Return True if m has a prime factor > limit."""
        if m == 1:
            return False
        for p in _STRIPE_PRIMES:
            if p > limit:
                break
            if m % p == 0:
                return _has_large_factor(m // p)
        if m <= limit:
            return False
        if is_prime(m):
            return m > limit
        d = _pollard_brent(m)
        if _has_large_factor(d):
            return True
        return _has_large_factor(m // d)
    return not _has_large_factor(n)

# ── Main construct() ─────────────────────────────────────────────────────────

def construct():
    start = time.time()
    deadline = start + 130   # 20 s safety margin before 150 s wall-clock kill

    best_k = 0
    best_p = 2

    # Phase 1: sieve small primes up to 5×10^6, check each
    SIEVE_LIMIT = 5_000_000

    def sieve_upto(limit):
        size = limit + 1
        comp = bytearray(size)
        for i in range(2, int(math.isqrt(limit)) + 1):
            if comp[i] == 0:
                step = i
                start_sq = i * i
                comp[start_sq::step] = b'\x01' * ((limit - start_sq) // step + 1)
        return comp

    comp = sieve_upto(SIEVE_LIMIT)

    for n_candidate in range(2, min(SIEVE_LIMIT + 1, 10**9)):
        if time.time() > deadline:
            break
        if comp[n_candidate] == 0:  # prime
            p = n_candidate
            base = p * p
            k = 0
            i = 1
            while i <= 200:
                if time.time() > deadline:
                    break
                val = base + i
                if not all_prime_factors_leq(val, p):
                    break
                k += 1
                i += 1
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 2: search large primes near 10^9 (descending) — larger p gives more room
    # Try odd candidates from 10^9 downward
    for offset in range(0, 5_000_000, 2):
        if time.time() > deadline:
            break
        c = 1_000_000_000 - offset
        if c <= SIEVE_LIMIT or c <= 2:
            continue
        if c % 2 == 0:
            c -= 1
        if not is_prime(c):
            continue
        p = c
        base = p * p
        k = 0
        i = 1
        while i <= 200:
            if time.time() > deadline:
                break
            val = base + i
            if not all_prime_factors_leq(val, p):
                break
            k += 1
            i += 1
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good one, we can try to extend the search
            # but keep going for now

    # Phase 3: random primes in mid-range
    for _ in range(2000):
        if time.time() > deadline:
            break
        c = random.randint(SIEVE_LIMIT, 10**9)
        if c % 2 == 0:
            c += 1
        if c > 10**9:
            c -= 2
        if not is_prime(c):
            continue
        p = c
        base = p * p
        k = 0
        i = 1
        while i <= 200:
            if time.time() > deadline:
                break
            val = base + i
            if not all_prime_factors_leq(val, p):
                break
            k += 1
            i += 1
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

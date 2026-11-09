"""Combined program: sieve-based prime search + fast smoothness check via trial division + Pollard-Brent."""
import time
import math
import random

_DEADLINE = 145.0  # seconds; keep margin before 150s kill

# Deterministic Miller-Rabin bases for n < 3.3e24
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(n):
    """Deterministic Miller-Rabin primality test for 64-bit integers."""
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


def _pollard_brent(n):
    """Pollard's Brent cycle-finding factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = random.randrange(1, 100)
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
                    q = (q * abs(x - y)) % n
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


def _sieve_primes(limit):
    """Generate list of all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]


def _is_p_smooth(n, p, small_primes):
    """
    Check if all prime factors of n are <= p.
    Uses trial division by small primes, then Miller-Rabin, then Pollard-Brent for remainder.
    """
    if n <= 1:
        return True
    if n <= p:
        return True

    # Trial division by small primes
    for sp in small_primes:
        if sp > p:
            break
        while n % sp == 0:
            n //= sp
        if n == 1:
            return True
        if n <= p:
            return True

    # If remaining n is prime, check if <= p
    if _is_prime(n):
        return n <= p

    # n is composite; factor with Pollard-Brent and check both parts
    d = _pollard_brent(n)
    if d == n:
        return n <= p
    if d > p:
        return False
    if n // d > p:
        return False
    return _is_p_smooth(d, p, small_primes) and _is_p_smooth(n // d, p, small_primes)


def _check_run(p, small_primes, max_k=200):
    """
    Check how many consecutive integers p^2+1, p^2+2, ..., p^2+k
    have all prime factors <= p. Return the maximum such k (0 to max_k).
    """
    base = p * p
    k = 0
    for i in range(1, max_k + 1):
        v = base + i
        if _is_p_smooth(v, p, small_primes):
            k = i
        else:
            break
    return k


def construct():
    t0 = time.time()
    best_k = 0
    best_p = 2

    # Precompute small primes for trial division - up to 100000
    SIEVE_LIMIT = 100000
    small_primes = _sieve_primes(SIEVE_LIMIT)

    # Phase 1: Exhaustive search over primes up to 5,000,000
    search_limit = 5000000
    search_primes = _sieve_primes(search_limit)

    for p in search_primes:
        if time.time() - t0 > _DEADLINE - 20:
            break
        if p < 2:
            continue
        k = _check_run(p, small_primes)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # Phase 2: Segmented sieve for larger primes up to 10^8
    if best_k < 200 and time.time() - t0 < _DEADLINE - 30:
        window_size = 500000
        for start in range(search_limit, 100000000, window_size):
            if time.time() - t0 > _DEADLINE - 20:
                break
            end = min(start + window_size, 100000000)
            sieve = bytearray(b'\x01') * (end - start)
            for sp in small_primes:
                if sp * sp > end:
                    break
                first = ((start + sp - 1) // sp) * sp
                if first < sp * sp:
                    first = sp * sp
                for j in range(first - start, end - start, sp):
                    sieve[j] = 0
            for i in range(end - start):
                if time.time() - t0 > _DEADLINE - 20:
                    break
                cand = start + i
                if cand < 2:
                    continue
                if sieve[i] and _is_prime(cand):
                    k = _check_run(cand, small_primes)
                    if k > best_k:
                        best_k = k
                        best_p = cand
                        if best_k >= 200:
                            break

    # Phase 3: Random search for larger primes if time remains
    if best_k < 200 and time.time() - t0 < _DEADLINE - 15:
        attempts = 0
        while time.time() - t0 < _DEADLINE - 5 and attempts < 50000:
            p = random.randrange(1000, 10**9, 2)
            if _is_prime(p):
                k = _check_run(p, small_primes)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break
            attempts += 1

    return {'p': best_p, 'k': best_k}

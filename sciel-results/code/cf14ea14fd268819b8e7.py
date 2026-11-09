"""Rewritten program: fast primality + smoothness check, search primes up to deadline."""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = 115.0  # seconds; keep margin before 120s kill

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime(n):
    """Deterministic Miller-Rabin primality test for 64-bit integers."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n-1 = d * 2^r
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
        # restart with new y, c

def _largest_prime_factor(n, limit):
    """
    Return the largest prime factor of n, or None if all factors are <= limit.
    Uses trial division by small primes, then Pollard-Brent for the remainder.
    If at any point a factor > limit is found, return it immediately.
    """
    if n <= limit:
        return None
    # Trial division by small primes up to sqrt(n) but stop early
    for p in _SMALL_PRIMES:
        if p > limit:
            break
        while n % p == 0:
            n //= p
        if n <= limit:
            return None
    # Now n has no small prime factors (up to 37) and n > limit
    # Use Pollard-Brent
    factors = []
    stack = [n]
    while stack:
        v = stack.pop()
        if v <= 1:
            continue
        if _is_prime(v):
            if v > limit:
                return v
            continue
        # Find a factor
        f = _pollard_brent(v)
        if f == v:
            # Should not happen, but guard
            if v > limit:
                return v
            continue
        stack.append(f)
        stack.append(v // f)
    return None

def _check_run(p, max_k=200):
    """
    Check how many consecutive integers p^2+1, p^2+2, ..., p^2+k
    have all prime factors <= p. Return the maximum such k (0 to max_k).
    """
    base = p * p
    k = 0
    for i in range(1, max_k + 1):
        v = base + i
        # Quick check: if v <= p, then trivially all factors <= p
        if v <= p:
            k = i
            continue
        # Check if v has any prime factor > p
        lp = _largest_prime_factor(v, p)
        if lp is not None:
            break
        k = i
    return k

def construct():
    t0 = time.time()
    best_k = 0
    best_p = 2

    # Small primes are easy to verify
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if time.time() - t0 > _DEADLINE - 5:
            break
        k = _check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes
    # Start from 41
    p = 41
    while True:
        if time.time() - t0 > _DEADLINE - 10:
            break
        if _is_prime(p):
            k = _check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
                # If we hit the max, we can stop
                if best_k >= 200:
                    break
        p += 2

    # If we have time, try a random search for larger primes
    if time.time() - t0 < _DEADLINE - 15:
        # Random search in a wider range
        attempts = 0
        while time.time() - t0 < _DEADLINE - 5 and attempts < 50000:
            # Random odd number up to 10^9
            p = random.randrange(1000, 10**9, 2)
            if _is_prime(p):
                k = _check_run(p)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break
            attempts += 1

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END

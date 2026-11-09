"""Rewritten program: efficient search for p with long p-smooth run after p^2."""
import time
import random
import math
import sys

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
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

def pollard_brent(n):
    """Pollard's Brent's cycle-finding factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = 128
        g = 1
        r = 1
        q = 1
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

def factorize(n):
    """Full factorization of n, returning list of prime factors (with multiplicity)."""
    if n <= 1:
        return []
    factors = []
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if is_prime_mr(n):
        factors.append(n)
        return factors
    # Use Pollard-Brent recursively
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime_mr(m):
            factors.append(m)
            continue
        d = pollard_brent(m)
        stack.append(d)
        stack.append(m // d)
    return factors

def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n <= 1:
        return 1
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    max_f = 1
    for p in small_primes:
        while n % p == 0:
            max_f = max(max_f, p)
            n //= p
    if n == 1:
        return max_f
    if is_prime_mr(n):
        return max(max_f, n)
    # Factor the remainder
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime_mr(m):
            max_f = max(max_f, m)
            continue
        d = pollard_brent(m)
        stack.append(d)
        stack.append(m // d)
    return max_f

def run_length(p, max_k=200, deadline=None):
    """Compute the run length k for prime p."""
    k = 0
    p2 = p * p
    for i in range(1, max_k + 1):
        if deadline is not None and time.time() > deadline:
            return k
        n = p2 + i
        if largest_prime_factor(n) <= p:
            k += 1
        else:
            break
    return k

def sieve_primes(limit):
    """Sieve of Eratosthenes up to limit."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_k = 0
    best_p = 3

    # Phase 1: Sieve small primes up to 100000 for quick screening
    small_limit = 100000
    small_primes = sieve_primes(small_limit)

    # Phase 2: Check primes in increasing order up to 100000
    for p in small_primes:
        if time.time() > deadline - 10:
            break
        if p > 100000:
            break
        # Quick check: if p is small, the run is likely short
        # But still check for correctness
        k = run_length(p, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break

    # Phase 3: Check larger primes in random order, focusing on primes > 100000
    # Use a larger sieve for primes up to 10^6, then random larger primes
    large_limit = 1000000
    large_primes = sieve_primes(large_limit)

    # Check primes from 100000 to 1000000 in random order (larger primes more likely to have longer runs)
    candidates = [p for p in large_primes if p > small_limit]
    random.shuffle(candidates)

    for p in candidates:
        if time.time() > deadline - 5:
            break
        k = run_length(p, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break

    # Phase 4: Random large primes up to 10^9
    # Generate random odd numbers and test primality
    random.seed(42)
    checked = set()
    while time.time() < deadline - 2:
        p = random.randrange(1000000, 1000000000, 2) | 1  # odd number
        if p in checked:
            continue
        checked.add(p)
        if not is_prime_mr(p):
            continue
        k = run_length(p, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break

    return {'p': best_p, 'k': best_k}

"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses Pollard-Brent + deterministic Miller-Rabin for fast factorization, and searches
primes efficiently via sieve to maximize k within the time budget."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard-Brent (faster than basic Rho) ----------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
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
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

# ---------- Factorization ----------
def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_brent(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # First try small primes for quick rejection
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    # Use Pollard-Brent for the rest
    factors = factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = 0
    is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to 2000000 for efficient search
    sieve_limit = 2000000
    primes = sieve_primes(sieve_limit)
    
    # Try each prime in the sieve
    for p in primes:
        if time.time() > deadline:
            break
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, try larger primes using random sampling + Miller-Rabin
    # Larger primes are more likely to have longer smooth runs
    if time.time() < deadline - 5:
        # Try random large primes
        attempts = 0
        while time.time() < deadline - 3 and attempts < 200:
            attempts += 1
            # Generate a random odd number in a range
            # Focus on ranges where primes are denser relative to factorization cost
            lo = random.randint(10**6, 10**9)
            hi = min(lo + 10000, 10**9)
            candidate = random.randint(lo, hi)
            if candidate % 2 == 0:
                candidate += 1
            # Check primality
            if not is_prime(candidate):
                continue
            p = candidate
            if p <= best_p:
                continue
            k = 0
            p_sq = p * p
            while k < 200:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END

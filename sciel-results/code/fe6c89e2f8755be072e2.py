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
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Strategy: Focus on medium-to-large primes where longer smooth runs are more likely
    # Small primes can't have long runs because p^2+i quickly exceeds what's p-smooth
    
    # Phase 1: Sieve primes in a medium range (10^5 to 10^7)
    # These are large enough to potentially have long runs but small enough to factor quickly
    sieve_lo = 100000
    sieve_hi = 10000000
    primes = sieve_primes(sieve_hi)
    
    # Filter to our range
    primes = [p for p in primes if sieve_lo <= p <= sieve_hi]
    
    # Sort by descending p - larger primes more likely to have longer runs
    # But also interleave some smaller ones for diversity
    primes_desc = sorted(primes, reverse=True)
    
    # Try primes in a mixed order: mostly large, some medium
    test_order = []
    for i, p in enumerate(primes_desc):
        if time.time() > deadline - 5:
            break
        test_order.append(p)
        # Every 5th, add a medium prime for diversity
        if i % 5 == 4 and i < len(primes_desc) - 5:
            test_order.append(primes_desc[len(primes_desc) // 2 + i // 5])
    
    # Test each prime
    for p in test_order:
        if time.time() > deadline - 3:
            break
        
        # Quick pre-filter: if p is too small relative to what we need, skip
        # For k=200, we need p^2+200 to be p-smooth, which requires p >= ~sqrt(200) ~ 15
        # But for longer runs, we want larger p
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline - 2:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: If time remains, try random large primes (10^7 to 10^9)
    if time.time() < deadline - 10:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 500:
            attempts += 1
            # Generate a random odd number in a range
            # Focus on ranges where primes are denser relative to factorization cost
            lo = random.randint(10**7, 10**9)
            hi = min(lo + 100000, 10**9)
            candidate = random.randint(lo, hi)
            if candidate % 2 == 0:
                candidate += 1
            # Check primality
            if not is_prime(candidate):
                continue
            p = candidate
            if p <= best_p and best_k >= 50:
                continue
            k = 0
            p_sq = p * p
            while k < 200:
                if time.time() > deadline - 3:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}

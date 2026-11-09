"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses segmented sieve for prime generation, optimized Pollard-Brent for factorization,
and searches primes efficiently to maximize k within the time budget."""

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

# ---------- Pollard's Rho with Brent's improvement ----------
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        q = 1
        r = 1
        m = 128
        ys = y
        while d == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and d == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                d = math.gcd(q, n)
                k += m
            r *= 2
        if d != n:
            return d
        d = 1
        while d == 1:
            ys = (ys * ys + c) % n
            d = math.gcd(abs(x - ys), n)
        if d != n:
            return d

# ---------- Factorization ----------
def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def all_factors_at_most_p(n, p, trial_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    for q in trial_primes:
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if n > p and is_prime(n):
        return False
    factors = factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def segmented_sieve(limit):
    """Generate all primes up to limit using segmented sieve."""
    if limit < 2:
        return []
    # First segment
    size = 10000
    primes = []
    # Generate small primes first with simple sieve
    small_limit = int(math.isqrt(limit)) + 1
    small_sieve = bytearray(small_limit + 1)
    small_sieve[0] = small_sieve[1] = 1
    for i in range(2, math.isqrt(small_limit) + 1):
        if not small_sieve[i]:
            for j in range(i * i, small_limit + 1, i):
                small_sieve[j] = 1
    small_primes = [i for i in range(2, small_limit + 1) if not small_sieve[i]]
    primes.extend(small_primes)
    
    # Segmented sieve for the rest
    start = small_limit + 1
    while start <= limit:
        end = min(start + size - 1, limit)
        seg = bytearray(end - start + 1)
        for p in small_primes:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first == p:
                first += p
            for j in range(first - start, end - start + 1, p):
                seg[j] = 1
        for i in range(end - start + 1):
            if not seg[i] and start + i > 1:
                primes.append(start + i)
        start = end + 1
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute trial division primes up to 5000
    trial_primes = segmented_sieve(5000)
    
    # Check p=2
    k = 0
    while k < 200 and all_factors_at_most_p(4 + k + 1, 2, trial_primes):
        k += 1
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Strategy: Use segmented sieve to generate primes, then check them
    # For larger p, the condition is easier to satisfy
    # Phase 1: Check primes up to 100000
    # Phase 2: Check primes up to 1000000
    # Phase 3: Check primes up to 10000000 if time permits
    
    # We'll search in increasing order but prioritize checking more values for larger p
    # since larger p makes the condition easier
    
    # Generate primes up to 100000 first
    primes_100k = segmented_sieve(100000)
    primes_100k = [p for p in primes_100k if p > 2]
    
    for p in primes_100k:
        if time.time() > deadline:
            break
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if not all_factors_at_most_p(p_sq + k + 1, p, trial_primes):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Generate primes up to 1000000
    if time.time() < deadline - 30:
        primes_1m = segmented_sieve(1000000)
        primes_1m = [p for p in primes_1m if p > 100000]
        
        for p in primes_1m:
            if time.time() > deadline:
                break
            
            k = 0
            p_sq = p * p
            while k < 200:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p, trial_primes):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Generate primes up to 10000000 if we still have time
    if time.time() < deadline - 20:
        primes_10m = segmented_sieve(10000000)
        primes_10m = [p for p in primes_10m if p > 1000000]
        
        for p in primes_10m:
            if time.time() > deadline:
                break
            
            k = 0
            p_sq = p * p
            while k < 200:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p, trial_primes):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # If we still have time, try some larger primes via Miller-Rabin
    if time.time() < deadline - 15:
        # Try random large primes
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            p = random.randrange(10**7, 10**9, 2)
            if not is_prime(p):
                continue
            
            k = 0
            p_sq = p * p
            while k < 200:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p, trial_primes):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}

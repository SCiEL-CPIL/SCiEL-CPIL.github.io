"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses optimized Pollard's rho + Miller-Rabin for fast factorization, and searches
primes efficiently to maximize k within the time budget."""

import time
import random
import math
import sys

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return n == 2 or n == 3
    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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
        # Brent's cycle detection
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
        # If d == n, restart with different parameters
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

# Precompute small primes for trial division
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # First try small primes
    for q in _SMALL_PRIMES:
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    # Quick check: if n > p and is_prime(n), then n is a prime factor > p
    if n > p:
        if is_prime(n):
            return False
        # n is composite and > p, need to factor it
        factors = factor(n)
        for f in factors:
            if f > p:
                return False
        return True
    # n <= p, so all factors are <= p
    return True

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, limit + 1, i):
                is_p[j] = False
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Phase 0: Try p=2 first
    k = 0
    while k < 200 and all_factors_at_most_p(4 + k + 1, 2):
        k += 1
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Generate primes up to 100000 using sieve
    primes_100k = sieve_primes(100000)
    
    # Phase 1: Check all primes up to 100000
    for p in primes_100k:
        if time.time() > deadline:
            break
        if p == 2:
            continue  # Already checked
        
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
    
    # Phase 2: Check primes from 100000 to 1000000 using a larger sieve
    if time.time() < deadline - 5:
        primes_1m = sieve_primes(1000000)
        # Only check primes > 100000
        for p in primes_1m:
            if time.time() > deadline:
                break
            if p <= 100000:
                continue
            
            k = 0
            p_sq = p * p
            # For larger p, limit the check to save time
            max_check = min(200, 100)
            while k < max_check:
                if time.time() > deadline:
                    break
                if not all_factors_at_most_p(p_sq + k + 1, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: If we still have time, try some larger primes
    # Use a segmented approach or just try specific ranges
    if time.time() < deadline - 10:
        # Try primes in ranges around powers of 10
        ranges_to_try = [(10**6, 2*10**6), (2*10**6, 5*10**6), (5*10**6, 10**7)]
        
        for low, high in ranges_to_try:
            if time.time() > deadline - 5:
                break
            
            # Generate primes in this range using a simple trial division approach
            # This is slower but we only do it if we have time
            for p in range(low, high, 2):
                if time.time() > deadline:
                    break
                if not is_prime(p):
                    continue
                
                k = 0
                p_sq = p * p
                max_check = min(200, 50)
                while k < max_check:
                    if time.time() > deadline:
                        break
                    if not all_factors_at_most_p(p_sq + k + 1, p):
                        break
                    k += 1
                
                if k > best_k:
                    best_k = k
                    best_p = p
                
                # Limit the number of primes we check per range to avoid timeout
                if (p - low) > 50000:
                    break
    
    return {'p': best_p, 'k': best_k}

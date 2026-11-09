"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a direct approach: search for primes p and check consecutive integers p^2+1, ..., p^2+k
for p-smoothness (all prime factors <= p). Uses efficient trial division and Pollard-Brent
with deterministic Miller-Rabin for factorization. Maximizes k within time budget."""

import time
import math
import random

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

# ---------- Pollard's Brent ----------
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

# ---------- Factorization with early termination for smoothness check ----------
def is_p_smooth(n, p, small_primes=None):
    """Check if all prime factors of n are <= p. Returns True if n is 1 or p-smooth."""
    if n <= 1:
        return True
    
    # Try small primes first with trial division
    if small_primes is None:
        small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for q in small_primes:
        if q > p:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Use Pollard-Brent to find a factor, then recursively check
    factors = []
    _factor(n, factors, p)
    for f in factors:
        if f > p:
            return False
    return True

def _factor(n, factors, p):
    """Recursively factor n, adding factors to list. Early termination if factor > p."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    _factor(d, factors, p)
    _factor(n // d, factors, p)

# Precompute small primes for fast trial division
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499)

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit using sieve
    sieve_limit = 20_000_000
    primes = sieve_primes(sieve_limit)
    
    # Precompute for each small prime q <= 1000, the set of residues r mod q
    # such that r^2 + j ≡ 0 (mod q) for some j in [1, 200].
    # A prime p is eliminated if p mod q is in this set AND q > p.
    
    precomputed_bad = {}
    small_primes_for_sieve = [q for q in primes if q <= 1000]
    for q in small_primes_for_sieve:
        bad_residues = set()
        for j in range(1, 201):
            target = (-j) % q
            # Find all r such that r^2 ≡ target (mod q)
            for r in range(q):
                if (r * r) % q == target:
                    bad_residues.add(r)
        precomputed_bad[q] = bad_residues
    
    # For each prime p, use precomputed bad residues to quickly eliminate candidates
    for idx, p in enumerate(primes):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Skip very small primes
        if p < 100:
            continue
        
        # Quick pre-filter: check if p is eliminated by any small prime q > p
        eliminated = False
        for q in small_primes_for_sieve:
            if q <= p:
                continue  # q must be > p
            if p % q in precomputed_bad[q]:
                eliminated = True
                break
        
        if eliminated:
            continue
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() - start_time > deadline:
                break
            n = p_sq + k + 1
            if not is_p_smooth(n, p, _SMALL_PRIMES):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, try some larger primes
    if time.time() - start_time < deadline - 30:
        for _ in range(500):
            if time.time() - start_time > deadline - 5:
                break
            
            lo = max(sieve_limit + 1, 10**6)
            hi = min(10**9, 5*10**8)
            candidate = random.randrange(lo, hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            while not is_prime(candidate):
                candidate += 2
                if candidate > hi:
                    break
            
            if candidate > hi:
                continue
            
            k = 0
            p_sq = candidate * candidate
            while k < 200:
                if time.time() - start_time > deadline - 1:
                    break
                n = p_sq + k + 1
                if not is_p_smooth(n, candidate, _SMALL_PRIMES):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}

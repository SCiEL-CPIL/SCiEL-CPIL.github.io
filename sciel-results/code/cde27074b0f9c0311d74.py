"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
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

def pollard_brent(n):
    """Brent's variant of Pollard's rho - faster than Floyd's."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, 100)
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
                if g > 1:
                    break
        if g != n:
            return g

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

def sieve_primes(limit):
    """Generate list of all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    
    v = n
    # Trial divide by small primes
    # Limit is min(p, sqrt(n)) but we only have small_primes up to 100000
    # If p > 100000, we divide by all small_primes.
    # If p < 100000, we divide by primes <= p.
    
    sqrt_n = int(math.isqrt(n))
    limit = min(p, sqrt_n)
    
    for sp in small_primes:
        if sp > limit:
            break
        while v % sp == 0:
            v //= sp
            
    if v == 1:
        return True
        
    if is_prime(v):
        return v <= p
        
    # v is composite.
    # If we divided out all primes <= limit, and v is composite,
    # it has a factor > limit.
    # If limit == p, then factor > p, so not p-smooth.
    # If limit == sqrt_n < p, then v must be 1 or prime (since any composite v has factor <= sqrt(v) <= sqrt(n) = limit).
    # So if v is composite here, it implies limit < sqrt(v) which is impossible if we checked up to sqrt_n.
    # Wait, if limit = p < sqrt_n, we stopped at p.
    # v has no factors <= p.
    # If v is composite, it has a factor q > p.
    # So not p-smooth.
    
    return False

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    # Use primes up to 100000 for fast trial division
    small_primes = sieve_primes(100000)
    
    # We will search primes in increasing order.
    # Based on knowledge, 1M-10M is a good range.
    # We'll sieve up to 10^8 if time permits, but start with 10^6.
    
    max_limit = 100000000  # 10^8
    
    # Segmented sieve to iterate through primes
    window_size = 1000000
    start = 2
    
    while start < max_limit and time.time() < deadline:
        end = min(start + window_size, max_limit)
        
        # Sieve this window
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
            if time.time() > deadline:
                break
            cand = start + i
            if cand < 2:
                continue
            if sieve[i] and is_prime(cand):
                k = 0
                base = cand * cand
                for ii in range(1, 201):
                    n = base + ii
                    if is_p_smooth(n, cand, small_primes):
                        k += 1
                    else:
                        break
                
                if k > best_k:
                    best_k = k
                    best_p = cand
                    
        start = end
        
    if best_p == 0:
        best_p = 2
        best_k = 1
        
    return {'p': best_p, 'k': best_k}

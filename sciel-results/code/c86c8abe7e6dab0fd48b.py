import math
import time
import random

# Build a pre-sieved set of candidate p using a sieve up to a modest bound,
# then extend with random / structured candidates up to 1e9 if time allows.
# Use a sieve up to SIEVE_LIMIT for fast primality.
SIEVE_LIMIT = 200000

# Global sieve cache
def _build_sieve(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = b'\x00' * (len(is_p[i*i::i]))
    return is_p

# Deterministic Miller-Rabin for 64-bit ints
def is_prime_mr(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = 128
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

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def factorize_with_limit(n, limit):
    """Return list of factors, or None if a factor exceeds limit."""
    factors = []
    for p in _SMALL_PRIMES:
        if p > limit:
            # If remaining n has a factor, the smallest factor <= sqrt(n);
            # but if n itself (if prime) is <= limit, we are OK.
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    # If n <= limit, it may be prime or composite with factors <= limit
    if n <= limit:
        # Factor n fully (all factors must be <= n <= limit)
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return factors
    # n > limit: we need to ensure no factor > limit.
    # If n is prime, it's > limit -> fail.
    # If n is composite, check if it has all factors <= limit.
    # But since n > limit, if it has all factors <= limit, then n = product of factors each <= limit.
    # That is possible, e.g., n = 2 * 3 * 5 * ... But n could also have a large prime factor > limit.
    # Check: factor n, verify all <= limit.
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime_mr(m):
            if m > limit:
                return None
            factors.append(m)
        else:
            d = pollard_brent(m)
            if d == m:
                stack.append(m)
            else:
                stack.append(d)
                stack.append(m // d)
    return factors

def _check_run(p, k_max, deadline):
    """Check the run length starting at p^2+1, up to k_max or until deadline exceeded."""
    k = 0
    p2 = p * p
    while k < k_max:
        if time.time() >= deadline:
            break
        v = p2 + k + 1
        # Factorize and check all factors <= p
        factors = factorize_with_limit(v, p)
        if factors is None:
            break
        k += 1
    return k

def _segmented_smoothness_check(p, k_max, deadline):
    """
    Use segmented sieve to check p-smoothness of p^2+1, ..., p^2+k_max.
    Returns the longest run length k such that all numbers p^2+1..p^2+k are p-smooth.
    
    Strategy:
    1. Generate all primes <= p (or up to a reasonable bound if p is very large)
    2. For each prime q <= p, mark multiples of q in the range [p^2+1, p^2+k_max]
    3. A number is p-smooth if all its prime factors are <= p, which means
       after dividing out all primes <= p, the remainder must be 1.
    
    More efficient approach:
    - Create an array rem[0..k_max-1] initialized to p^2+1, ..., p^2+k_max
    - For each prime q <= p, divide out all factors of q from each rem[i]
    - After processing all primes <= p, rem[i] == 1 iff p^2+i+1 is p-smooth
    """
    # We need primes up to p. If p is large, we can't sieve up to p.
    # But we only need to divide by primes q where q^2 <= max_value in range,
    # and also handle the case where rem[i] is prime > p.
    # Actually, for p-smoothness: after removing all prime factors <= p,
    # the remainder must be 1. If remainder > 1, it has a prime factor > p.
    
    # Key insight: if p is large (say > 10^6), we can't sieve up to p.
    # But we can use a hybrid: sieve up to some bound B, then for remaining
    # composites, use trial division or Pollard.
    
    # For the segmented approach to work efficiently, we need primes up to p.
    # If p <= SIEVE_LIMIT, we can use the pre-built sieve.
    # If p > SIEVE_LIMIT, we need a different strategy.
    
    # Let's handle two cases:
    # Case 1: p <= SIEVE_LIMIT (small p) - use full sieve
    # Case 2: p > SIEVE_LIMIT (large p) - use hybrid approach
    
    if p <= SIEVE_LIMIT:
        # Get all primes <= p from sieve
        is_prime_sieve = _build_sieve(p)
        primes = [i for i in range(2, p + 1) if is_prime_sieve[i]]
    else:
        # For large p, we can't sieve up to p. 
        # Instead, we'll use trial division by small primes, then check if remainder is 1 or prime > p.
        # This is less efficient but works.
        # Actually, let's use a different approach for large p:
        # Just check each number individually with the factorize_with_limit function,
        # but that's what we're trying to avoid.
        # 
        # Alternative: for large p, p-smoothness is easier to satisfy because
        # p is large, so more prime factors are allowed.
        # We can use a quick check: if n < p^2, then any prime factor of n is < p^2.
        # But we need factors <= p.
        # 
        # Let's just fall back to individual checking for large p, but with optimization.
        k = 0
        p2 = p * p
        while k < k_max:
            if time.time() >= deadline:
                break
            v = p2 + k + 1
            factors = factorize_with_limit(v, p)
            if factors is None:
                break
            k += 1
        return k
    
    # Now we have all primes <= p
    k = k_max
    p2 = p * p
    
    # Create remainder array
    rem = [p2 + i + 1 for i in range(k)]
    
    # For each prime q <= p, divide out factors of q from each rem[i]
    for q in primes:
        if time.time() >= deadline:
            break
        for i in range(k):
            while rem[i] % q == 0:
                rem[i] //= q
    
    # Now check which rem[i] == 1 (p-smooth)
    run = 0
    for i in range(k):
        if rem[i] == 1:
            run += 1
        else:
            break
    
    return run

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    random.seed(42)
    
    best_k = 0
    best_p = 2
    
    # Sieve for quick primality up to SIEVE_LIMIT
    is_prime_sieve = _build_sieve(SIEVE_LIMIT)
    
    # Check p=2 first
    k2 = _segmented_smoothness_check(2, 200, deadline)
    if k2 > best_k:
        best_k = k2
        best_p = 2
    
    # Search primes from sieve in DESCENDING order (larger p first)
    # Larger p allows more prime factors, increasing probability of p-smooth runs
    for p in range(SIEVE_LIMIT, 2, -2):
        if time.time() >= deadline:
            break
        if is_prime_sieve[p]:
            k_val = _segmented_smoothness_check(p, 200, deadline)
            if k_val > best_k:
                best_k = k_val
                best_p = p
                # If we found a good run, continue searching
                if best_k >= 15:
                    pass
    
    # If time remains, search larger primes
    if time.time() < deadline:
        tried_p = set(range(2, SIEVE_LIMIT + 1))
        
        attempts = 0
        max_attempts = 5000
        while time.time() < deadline - 5.0 and attempts < max_attempts:
            attempts += 1
            # Random prime search in medium range
            p = random.randint(10**5, 10**6) | 1
            if p in tried_p:
                continue
            tried_p.add(p)
            if is_prime_mr(p):
                k_val = _segmented_smoothness_check(p, 200, deadline)
                if k_val > best_k:
                    best_k = k_val
                    best_p = p
    
    return {'p': best_p, 'k': best_k}

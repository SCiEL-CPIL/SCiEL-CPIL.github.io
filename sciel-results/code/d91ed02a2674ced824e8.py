"""Rewritten program using segmented sieve + Pollard-Brent verification."""
import math
import random

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    """Pollard's Brent cycle detection factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, 10000)
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
        # retry with different parameters

def factorize(n, limit=None):
    """
    Factorize n, returning a dict of prime->exponent.
    If limit is given, stop factoring once all remaining factors would exceed limit
    (i.e., we only care if all prime factors <= limit).
    Returns (factor_dict, all_le_limit) where all_le_limit is True iff every prime factor <= limit.
    """
    if n <= 1:
        return {}, True
    if limit is not None and n > 1 and n <= limit:
        # If n itself is <= limit, it could be prime or composite, but all its factors are <= n <= limit
        # We need to verify it's actually composed of primes <= limit, which is trivially true if n <= limit
        # But wait: if n is prime and n <= limit, that's fine. If n is composite, its factors are < n <= limit.
        # So yes, all prime factors are <= limit.
        return {n: 1}, True  # We don't need full factorization for the check
    
    factors = {}
    remaining = n
    
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > remaining:
            break
        while remaining % p == 0:
            factors[p] = factors.get(p, 0) + 1
            remaining //= p
            if limit is not None and p > limit:
                return factors, False
    
    if remaining == 1:
        return factors, True
    
    if limit is not None and remaining <= limit:
        # All remaining factors are <= remaining <= limit
        if is_prime_mr(remaining):
            factors[remaining] = factors.get(remaining, 0) + 1
        else:
            # It's composite but <= limit, so all its prime factors are < remaining <= limit
            # We can just mark it as a "big factor" that's still <= limit
            factors[remaining] = factors.get(remaining, 0) + 1
        return factors, True
    
    # Use Pollard-Brent for the remaining part
    def _factor_rec(m, limit_check):
        if m == 1:
            return
        if is_prime_mr(m):
            factors[m] = factors.get(m, 0) + 1
            if limit_check and m > limit_check:
                raise ValueError("factor exceeds limit")
            return
        d = pollard_brent(m)
        _factor_rec(d, limit_check)
        _factor_rec(m // d, limit_check)
    
    try:
        _factor_rec(remaining, limit)
    except ValueError:
        return factors, False
    return factors, True

def check_p_smooth(v, p):
    """Check if all prime factors of v are <= p."""
    if v <= 1:
        return True
    if v <= p:
        return True
    # Quick check: if v has a prime factor > p, it's not p-smooth
    # Use trial division by small primes first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                     101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    remaining = v
    for sp in small_primes:
        if sp > p:
            break
        if sp * sp > remaining:
            break
        while remaining % sp == 0:
            remaining //= sp
    if remaining == 1:
        return True
    if remaining <= p:
        return True
    # remaining > p, need to check if it has any prime factor > p
    # If remaining is prime and > p, then it has a prime factor > p -> not p-smooth
    if is_prime_mr(remaining):
        return remaining <= p
    # remaining is composite and > p. Need to factor it.
    # Use Pollard-Brent
    def _has_large_factor(m, p_limit):
        if m <= 1:
            return False
        if m <= p_limit:
            return False
        if is_prime_mr(m):
            return m > p_limit
        d = pollard_brent(m)
        return _has_large_factor(d, p_limit) or _has_large_factor(m // d, p_limit)
    
    return not _has_large_factor(remaining, p)

def construct():
    # Strategy:
    # 1. Generate candidate primes up to some limit
    # 2. For each prime p, check p^2+1, ..., p^2+k for p-smoothness
    # 3. Use segmented sieve to quickly eliminate numbers with small prime factors > p
    #    Actually, for p-smoothness, we need all prime factors <= p.
    #    A number v in [p^2+1, p^2+200] is p-smooth iff it has no prime factor > p.
    #    
    # Key insight: For v = p^2 + i where 1 <= i <= 200, if v has a prime factor q > p,
    # then q > p and q divides p^2 + i. Since q > p, we have q > sqrt(p^2) = p, 
    # so q cannot divide p. Thus gcd(q, p) = 1.
    #
    # Actually, let's think differently. We want to find p such that many consecutive
    # integers after p^2 are p-smooth.
    #
    # For small p, we can brute force. For larger p, we need smarter methods.
    #
    # Let's start with a reasonable search. The seed program found best k=2 for p < 2000.
    # Let's search over a wider range of primes and use efficient checking.
    
    # First, let's do a quick search over small primes to establish a baseline
    # Then try larger primes with the segmented sieve approach
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to some limit for trial division
    # For checking p-smoothness of numbers around p^2, we need primes up to p
    # But p can be up to 10^9, so we can't sieve all primes up to p for each p.
    
    # Strategy: For each candidate prime p, check k=1 to 200:
    # For each i in 1..200, check if p^2+i is p-smooth.
    # Stop when we find a non-p-smooth number.
    
    # To make this efficient, we precompute small primes up to, say, 10^6.
    # For numbers v = p^2 + i, we first divide out all small prime factors.
    # If the remaining part is 1 or <= p, then v is p-smooth (since any remaining composite
    # factor would have prime factors < remaining <= p, or remaining is prime <= p).
    # If remaining > p, we need to check if it's prime (if so, not p-smooth) or composite
    # with all factors <= p.
    
    # Let's generate small primes up to 10^6
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    # For candidate primes p, we'll test primes in increasing order
    # Start with small primes, then go larger
    
    # Test primes up to 10^5 first with full checking
    # Then sample larger primes
    
    def is_prime_simple(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Test candidate primes
    # For efficiency, we'll test primes in batches
    
    # Phase 1: Test all primes up to 10^5
    for p in range(2, 100000):
        if not is_prime_simple(p):
            continue
        k = 0
        for i in range(1, 201):
            v = p * p + i
            if not check_p_smooth(v, p):
                break
            k = i
        if k > best_k:
            best_k = k
            best_p = p
        # Early exit optimization: if k is already 200, that's the max
        if k == 200:
            break
    
    # Phase 2: Test some larger primes (sampled)
    # For larger p, p^2 is huge, so we need to be careful with computation
    # But check_p_smooth should still work
    
    # Let's test primes in ranges [10^5, 10^6], [10^6, 10^7], etc.
    # Sample every few primes to keep within time limit
    
    import time
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    for p in range(100001, 1000000, 2):
        if time.time() - start_time > time_limit:
            break
        if not is_prime_mr(p):
            continue
        k = 0
        for i in range(1, 201):
            v = p * p + i
            if not check_p_smooth(v, p):
                break
            k = i
        if k > best_k:
            best_k = k
            best_p = p
        if k == 200:
            break
    
    # Phase 3: Test even larger primes if time permits
    if time.time() - start_time < time_limit:
        for p in range(1000001, 10000000, 1000):
            if time.time() - start_time > time_limit:
                break
            if not is_prime_mr(p):
                continue
            k = 0
            for i in range(1, 201):
                v = p * p + i
                if not check_p_smooth(v, p):
                    break
                k = i
            if k > best_k:
                best_k = k
                best_p = p
            if k == 200:
                break
    
    return {'p': best_p, 'k': best_k}

import time
import math
import random
import sys

def _sieve(limit):
    """Sieve of Eratosthenes, returns list of primes up to limit."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small_primes:
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
    """Pollard's Brent cycle detection for finding a factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    y = random.randint(1, n - 1)
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
    return g

def _is_p_smooth(n, p, primes):
    """Check if all prime factors of n are <= p.
    
    For n < 2*p^2 (which holds when n <= p^2 + k and k < p),
    any prime factor q of n satisfies: if q > p, then q >= next_prime_after_p.
    But since n < 2*p^2, if q > p then n = q * m with m < 2*p.
    We just need to verify no prime factor exceeds p.
    """
    if n <= 1:
        return True
    
    # Quick check: if n is prime
    if _is_prime_mr(n):
        return n <= p
    
    # Trial division by primes up to min(p, sqrt(n))
    v = n
    limit = min(p, math.isqrt(n))
    for q in primes:
        if q > limit:
            break
        while v % q == 0:
            v //= q
        if v == 1:
            return True
        # Early termination: if v is prime and <= p, done
        if v < q * q:
            break
    
    if v == 1:
        return True
    
    # v is composite or prime, all factors > limit >= min(p, sqrt(n))
    # If v is prime: check v <= p
    if _is_prime_mr(v):
        return v <= p
    
    # v is composite. Need to find a factor.
    # All prime factors of v are > limit.
    # If limit >= p, then all factors > p, so NOT p-smooth.
    if limit >= p:
        return False
    
    # limit < p means sqrt(n) < p, so v < p^2 and v is composite.
    # Try Pollard-Brent
    try:
        f = _pollard_brent(v)
    except Exception:
        return False
    return _is_p_smooth(f, p, primes) and _is_p_smooth(v // f, p, primes)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    # Precompute primes up to 10^7 for both search and trial division
    SIEVE_LIMIT = 10000000
    primes = _sieve(SIEVE_LIMIT)
    n_primes = len(primes)
    
    best_k = 0
    best_p = 2  # fallback
    
    # Phase 1: Search large primes first (most promising for long runs)
    # For large p, p^2+i < 2p^2 for i < p, so no prime factor can be in (p, 2p^2)
    # unless it's a product... actually the key is: any prime factor q of p^2+i
    # with q > p implies p^2+i = q*m where m < 2p (since p^2+i < 2p^2).
    # So we need to check that no such factorization exists, which trial division handles.
    
    # Search from large to small
    # Start from the largest primes we have
    for idx in range(n_primes - 1, -1, -1):
        if time.time() > deadline:
            break
        
        p = primes[idx]
        base = p * p
        
        k = 0
        # Check p^2+1, p^2+2, ...
        for i in range(1, 201):
            if time.time() > deadline:
                break
            n = base + i
            if _is_p_smooth(n, p, primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a very good k, we can stop early
            # but let's continue a bit more to be safe
            if k >= 50:
                # Found a good one, but keep searching a bit
                pass
        
        # Progress check: every 100k primes, check time
        if idx % 100000 == 0:
            if time.time() > deadline:
                break
    
    # Phase 2: If time remains, try random large primes beyond 10^7
    if time.time() < deadline - 20:
        random.seed(42)
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5000:
            attempts += 1
            # Generate random odd number in [10^7, 10^9]
            p_candidate = random.randint(10**7, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
            # Quick check with small primes
            is_cand = True
            for q in primes[:100]:
                if p_candidate % q == 0:
                    if p_candidate == q:
                        break
                    is_cand = False
                    break
            if not is_cand:
                continue
            if not _is_prime_mr(p_candidate):
                continue
            
            base = p_candidate * p_candidate
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                n = base + i
                if _is_p_smooth(n, p_candidate, primes):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p_candidate
    
    return {'p': best_p, 'k': best_k}

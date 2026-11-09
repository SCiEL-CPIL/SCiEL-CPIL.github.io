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
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if _is_prime_mr(n):
        return n <= p
    v = n
    limit = min(p, math.isqrt(n))
    for q in primes:
        if q > limit:
            break
        while v % q == 0:
            v //= q
        if v == 1:
            return True
        if v < q * q:
            break
    if v == 1:
        return True
    if _is_prime_mr(v):
        return v <= p
    if limit >= p:
        return False
    try:
        f = _pollard_brent(v)
    except Exception:
        return False
    return _is_p_smooth(f, p, primes) and _is_p_smooth(v // f, p, primes)

def _batch_check_p_smooth(base, length, p, primes):
    """Batched p-smoothness check for base+1, ..., base+length-1.
    
    For each prime q <= min(p, sqrt(base+length-1)), find all positions
    i in [1, length-1] where (base+i) is divisible by q, and divide out
    all factors of q. Then check if each remainder is 1 or a prime <= p.
    """
    remainders = [base + i for i in range(length)]
    
    max_val = base + length - 1
    sqrt_max = math.isqrt(max_val)
    max_q = min(p, sqrt_max)
    
    for q in primes:
        if q > max_q:
            break
        # Find smallest i >= 1 such that (base + i) % q == 0
        # i ≡ -base (mod q)
        r = (-base) % q
        if r < 1:
            r += q
        if r >= length:
            continue
        for i in range(r, length, q):
            while remainders[i] % q == 0:
                remainders[i] //= q
    
    k = 0
    for i in range(1, length):
        v = remainders[i]
        if v == 1:
            k += 1
        elif _is_prime_mr(v) and v <= p:
            k += 1
        else:
            break
    
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    SIEVE_LIMIT = 10000000
    primes = _sieve(SIEVE_LIMIT)
    n_primes = len(primes)
    
    best_k = 0
    best_p = 2
    
    for idx in range(n_primes - 1, -1, -1):
        if time.time() > deadline:
            break
        p = primes[idx]
        base = p * p
        k = _batch_check_p_smooth(base, 201, p, primes)
        if k > best_k:
            best_k = k
            best_p = p
    
    if time.time() < deadline - 30:
        random.seed(42)
        attempts = 0
        while time.time() < deadline - 10 and attempts < 2000:
            attempts += 1
            p_candidate = random.randint(10**7, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
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
            k = _batch_check_p_smooth(base, 201, p_candidate, primes)
            if k > best_k:
                best_k = k
                best_p = p_candidate
    
    return {'p': best_p, 'k': best_k}

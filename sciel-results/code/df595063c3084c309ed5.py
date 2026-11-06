"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_prime_mr(n):
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y, c, m = 2, 1, 128
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
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g
        c += 1

_SMALL_PRIMES = []
def _init_small_primes(limit=100000):
    global _SMALL_PRIMES
    if _SMALL_PRIMES:
        return
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    _SMALL_PRIMES = [i for i in range(2, limit + 1) if sieve[i]]

_init_small_primes(100000)

def _is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    for q in _SMALL_PRIMES:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    # n has all prime factors > 100000 (or > p if p < 100000)
    if n <= p:
        # n is prime > 100000 but <= p, so it's p-smooth
        return True
    # n > p: check if n is prime
    if _is_prime_mr(n):
        return False  # prime > p
    # n is composite, factor it
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime_mr(m):
            if m > p:
                return False
        else:
            d = _pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
    return True

def construct():
    start = time.time()
    deadline = start + 115
    
    best_k = 0
    best_p = 2
    
    p = 2
    while time.time() < deadline:
        if p == 2:
            p = 3
        else:
            p += 2
            while not _is_prime_mr(p):
                p += 2
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if p > 10**9:
            break
        
        # Quick reject: if p^2+1 is prime, k=0
        if _is_prime_mr(p * p + 1):
            continue
        
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            if _is_p_smooth(p * p + i, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END

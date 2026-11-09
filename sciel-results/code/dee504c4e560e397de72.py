"""Rewritten: segmented-sieve prime scan + fast p-smoothness check
(trial division by small primes, deterministic Miller-Rabin, Pollard-Brent).
Deadline-guarded to return best (p,k) before the 150s kill."""
import time
import math
import random

# ---------- Deterministic Miller-Rabin (n < 2^64) ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime_mr(n):
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
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard-Brent ----------
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
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
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
                if g > 1:
                    break
        if 1 < g < n:
            return g

# ---------- Small primes for trial division ----------
def _sieve_small(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    r = int(math.isqrt(limit))
    for i in range(2, r + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = bytearray(len(sieve[start:limit + 1:step]))
    return [i for i in range(limit + 1) if sieve[i]]

_SMALL = _sieve_small(100000)

def _all_factors_le(n, p):
    # n > 1, composite or unknown; ensure all prime factors <= p
    for sp in _SMALL:
        if sp > p or sp * sp > n:
            break
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
    if n == 1:
        return True
    if n <= p:
        return True
    if _is_prime_mr(n):
        return False
    d = _pollard_brent(n)
    for _ in range(8):
        if 1 < d < n:
            break
        d = _pollard_brent(n)
    if not (1 < d < n):
        return False
    return _all_factors_le(d, p) and _all_factors_le(n // d, p)

def _is_p_smooth(n, p):
    if n <= 1:
        return True
    for sp in _SMALL:
        if sp > p or sp * sp > n:
            break
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
    if n == 1:
        return True
    if n <= p:
        return True
    if _is_prime_mr(n):
        return False
    return _all_factors_le(n, p)

# ---------- Segmented sieve ----------
def _primes_iter(limit):
    r = int(math.isqrt(limit)) + 1
    small = _sieve_small(r)
    low = 2
    seg = 1 << 20
    while low <= limit:
        high = min(low + seg - 1, limit)
        buf = bytearray(b'\x01') * (high - low + 1)
        for sp in small:
            if sp * sp > high:
                break
            start = max(sp * sp, ((low + sp - 1) // sp) * sp)
            for j in range(start, high + 1, sp):
                buf[j - low] = 0
        for i, v in enumerate(buf):
            if v:
                yield low + i
        low = high + 1

# ---------- Main ----------
def construct():
    deadline = time.time() + 140.0
    best_p, best_k = 2, 1
    # Scan primes from small upward; small p factor faster, so we cover many.
    for p in _primes_iter(10**9):
        if time.time() > deadline - 4.0:
            break
        p2 = p * p
        k = 0
        for j in range(1, 201):
            if time.time() > deadline - 2.0:
                break
            if _is_p_smooth(p2 + j, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
    return {'p': best_p, 'k': best_k}

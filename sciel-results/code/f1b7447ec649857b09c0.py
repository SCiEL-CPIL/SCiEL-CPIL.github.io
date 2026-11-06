import time
import random

# Constants
DEADLINE = 110.0  # safety margin before 120s limit
SIEVE_LIMIT = 10**6

# Generate primes up to SIEVE_LIMIT using Sieve of Eratosthenes
def _sieve(n):
    is_p = bytearray([1]) * (n + 1)
    is_p[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]

_SMALL_PRIMES = _sieve(SIEVE_LIMIT)

def is_prime(n):
    """Deterministic Miller-Rabin for 64-bit integers."""
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
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

def pollard_brent(n):
    """Pollard's rho with Brent's cycle detection."""
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
                    q = (q * abs(x - y)) % n
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_p_smooth(n, p):
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
    if n > 1:
        return is_prime(n) and n <= p
    return True

def run_length(p):
    """Count how many consecutive integers p^2+1, p^2+2, ... are p-smooth."""
    k = 0
    base = p * p
    while True:
        k += 1
        if not is_p_smooth(base + k, p):
            return k - 1

def construct():
    t0 = time.time()
    best_k = 0
    best_p = None
    
    # Pre-compute small primes for trial division in is_p_smooth
    # Already done in _SMALL_PRIMES
    
    # Search primes p
    for p in _SMALL_PRIMES:
        if p >= SIEVE_LIMIT:
            break
        if time.time() - t0 > DEADLINE - 30:
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a very long run, we can be more aggressive
            if best_k >= 10:
                pass  # continue searching
    
    # If we have time, search larger primes
    if best_p is None:
        best_p = 3
    if time.time() - t0 < DEADLINE - 5:
        # Search primes from 10^6 to 10^9
        start = SIEVE_LIMIT + 1
        # Use a larger sieve for medium primes
        MEDIUM_SIEVE = 10**7
        if MEDIUM_SIEVE > SIEVE_LIMIT:
            medium_primes = _sieve(MEDIUM_SIEVE)
            # Skip primes already checked
            idx = 0
            while idx < len(medium_primes) and medium_primes[idx] < SIEVE_LIMIT:
                idx += 1
            for p in medium_primes[idx:]:
                if time.time() - t0 > DEADLINE - 30:
                    break
                k = run_length(p)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    # Search very large primes randomly
    if time.time() - t0 < DEADLINE - 5:
        random.seed(42)
        for _ in range(10000):
            if time.time() - t0 > DEADLINE - 30:
                break
            # Random odd number in [10^7, 10^9]
            p = random.randrange(10**7, 10**9, 2)
            if not is_prime(p):
                continue
            k = run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    # Final deadline check
    if time.time() - t0 > DEADLINE:
        # Emergency return
        if best_p is None:
            best_p = 3
            best_k = 1
        return {'p': best_p, 'k': max(1, best_k)}
    
    if best_p is None:
        best_p = 3
        best_k = 1
    
    return {'p': best_p, 'k': max(1, best_k)}

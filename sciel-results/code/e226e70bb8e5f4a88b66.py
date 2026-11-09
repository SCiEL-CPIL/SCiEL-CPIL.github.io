import time
import random
import math

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MILLER_RABIN_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MILLER_RABIN_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
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

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

def _is_p_smooth(n, p):
    """Return True if all prime factors of n are <= p."""
    if n <= 1:
        return True
    factors = []
    _factor(n, factors)
    return all(f <= p for f in factors)

# ---------- Sieve for small primes ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Pre-compute small primes for quick trial division
    small_primes = _sieve(10000)

    def _quick_factor_check(n, p, small_primes):
        """Quick check: divide by small primes, then Miller-Rabin/Pollard if needed."""
        t = n
        for q in small_primes:
            if q > p:
                break
            if q * q > t:
                break
            while t % q == 0:
                t //= q
        if t == 1:
            return True
        # If t > 1, check if t is prime and <= p, or composite with all factors <= p
        if _is_prime_mr(t):
            return t <= p
        # Need full factorization
        factors = []
        _factor(t, factors)
        return all(f <= p for f in factors)

    best_k = 0
    best_p = 3

    # Search small primes first (sieve up to 100000)
    primes_small = _sieve(100000)

    for p in primes_small:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            if _quick_factor_check(val, p, small_primes):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    # Search larger primes using random sampling
    # Try primes in ranges [10^5, 10^6], [10^6, 10^7], ..., [10^8, 10^9]
    log_ranges = [(5, 6), (6, 7), (7, 8), (8, 9)]
    attempts_per_range = 2000

    for log_lo, log_hi in log_ranges:
        if time.time() > deadline:
            break
        for _ in range(attempts_per_range):
            if time.time() > deadline:
                break
            # Random odd number in [10^log_lo, 10^log_hi)
            log_val = log_lo + random.random() * (log_hi - log_lo)
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            # Find next prime
            while not _is_prime_mr(p):
                p += 2
                if p >= 10**9:
                    break
            if p > 10**9 or not _is_prime_mr(p):
                continue

            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                if _quick_factor_check(val, p, small_primes):
                    k = i
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break

    return {'p': best_p, 'k': best_k}

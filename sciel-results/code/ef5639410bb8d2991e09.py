import math
import time

# EVOLVE-BLOCK-START

# ---------- Primality ----------
def _is_prime(n):
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin deterministic for n < 3.3e24 with these bases
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = 2, 1, 128
    g, q, r, x = 1, 1, 1, 0
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

def _factor(n, out):
    if n == 1:
        return
    if _is_prime(n):
        out.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, out)
    _factor(n // d, out)

# ---------- Sieve ----------
def _sieve(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # primes up to 2e6 for trial division in smoothness test
    SIEVE_LIMIT = 2000000
    primes = _sieve(SIEVE_LIMIT)

    # We need candidate primes p up to 1e9.
    # Strategy: for each candidate p, test p^2+1..p^2+k for p-smoothness.
    # A number n is p-smooth iff all its prime factors <= p.
    # Efficient test: trial divide by primes <= min(p, sqrt(n)); if remainder is 1 or (prime and <= p), smooth.
    # But n = p^2+i, sqrt(n) ~ p, so trial division up to p is O(pi(p)) ~ p/ln p. For p up to 1e9 that's too slow per number.
    # Better: factor n fully with Pollard-Brent and check max factor <= p.

    def largest_prime_factor(n):
        """Return largest prime factor of n, or 1 if n==1."""
        if n == 1:
            return 1
        # quick small-prime trial division
        m = n
        for q in primes:
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
        if m == 1:
            # all factors were in small primes; need the largest
            # recompute
            return max((q for q in primes if q * q <= n and n % q == 0), default=1) if False else _lpf_small(n)
        if _is_prime(m):
            return m
        # m is composite; factor it
        out = []
        _factor(m, out)
        return max(out)

    def _lpf_small(n):
        # n is small enough that all its prime factors are in our sieve
        best = 1
        m = n
        for q in primes:
            if q * q > m:
                break
            if m % q == 0:
                best = q
                while m % q == 0:
                    m //= q
        if m > 1:
            best = max(best, m)
        return best

    def is_p_smooth(n, p):
        """Return True iff all prime factors of n are <= p."""
        if n == 1:
            return True
        m = n
        # trial divide by small primes
        for q in primes:
            if q > p:
                # if we've divided by all primes <= p and m > 1, then m has a factor > p
                # but m might be prime > p, or composite with all factors > p
                if m > 1:
                    return False
                break
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
        # after trial division, m is either 1, or prime, or composite with factors > sqrt(original remaining)
        if m == 1:
            return True
        if m > p:
            return False
        # m <= p, check if m is prime (it is, since we divided all smaller primes)
        # actually m might be composite with factors <= p but we stopped at q*q > m
        # if q*q > m and m > 1, then m is prime
        return m <= p

    def run_length(p):
        """Count consecutive k such that p^2+1..p^2+k are all p-smooth."""
        k = 0
        base = p * p
        while k < 200:
            if time.time() > deadline:
                break
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        return k

    # Candidate primes: we can't sieve up to 1e9. Use a segment sieve or random search.
    # For now, search primes up to 5e6 with a larger sieve, then extend.
    # Actually, let's sieve up to 5e6 for candidates, and also try some larger primes via Miller-Rabin scan.
    CAND_LIMIT = 5000000
    cand_primes = _sieve(CAND_LIMIT)

    best_k = 0
    best_p = 2

    for p in cand_primes:
        if time.time() > deadline - 5:
            break
        if p * p > 10**18:  # not needed for p <= 1e9
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
        # early stop: if p is large, p^2+1 is likely prime -> k=0
        # but we can't easily predict; keep going within time budget

    # If time remains, try larger primes via scanning
    if time.time() < deadline - 10:
        # scan odd numbers from CAND_LIMIT+1 upward, test primality, then run_length
        start = CAND_LIMIT + 1
        if start % 2 == 0:
            start += 1
        n = start
        while n < 10**9 and time.time() < deadline - 5:
            if _is_prime(n):
                k = run_length(n)
                if k > best_k:
                    best_k = k
                    best_p = n
            n += 2

    if best_k < 1:
        best_k = 1
        # find a p where p^2+1 is p-smooth
        for p in cand_primes:
            if is_p_smooth(p * p + 1, p):
                best_p = p
                break

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END

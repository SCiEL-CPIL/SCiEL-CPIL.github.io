"""Rewritten program: fast Miller-Rabin + Pollard-Brent factoring, broader prime search."""
# EVOLVE-BLOCK-START

def construct():
    # ---------- Deterministic Miller-Rabin for 64-bit ----------
    def is_prime(n):
        if n < 2:
            return False
        # small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # write n-1 = d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # bases sufficient for n < 2^64
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

    # ---------- Pollard's Rho (Brent) ----------
    import math
    import random

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q = 1
            x = ys = y
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    def factor(n, result):
        if n == 1:
            return
        if is_prime(n):
            result.append(n)
            return
        d = pollard_brent(n)
        factor(d, result)
        factor(n // d, result)

    def largest_prime_factor(n):
        # quick small-prime trial division to speed up
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        temp = n
        for p in small_primes:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            return 37 if n > 1 else 1
        factors = []
        factor(temp, factors)
        return max(factors) if factors else 1

    # ---------- Generate primes up to some limit ----------
    # We'll search primes up to ~2_000_000. Use a simple sieve for speed.
    LIMIT = 2_000_000
    sieve = bytearray(b'\x01') * (LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(LIMIT ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    primes = [i for i in range(2, LIMIT + 1) if sieve[i]]

    # ---------- Search for best (k, p) ----------
    best_k = 0
    best_p = 2

    # We can afford to check many primes. Let's check all primes up to LIMIT.
    # For each prime p, check p^2+1, p^2+2, ... until we find one with a prime factor > p.
    # To stay within time, we limit k to 200 as per spec.

    for p in primes:
        # p must be at most 10^9; our primes are <= 2e6, so fine.
        p2 = p * p
        k = 0
        for offset in range(1, 201):
            val = p2 + offset
            # Check if all prime factors of val are <= p
            # Quick check: if val has a prime factor > p, stop.
            # We need the largest prime factor.
            lpf = largest_prime_factor(val)
            if lpf > p:
                break
            k = offset
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END

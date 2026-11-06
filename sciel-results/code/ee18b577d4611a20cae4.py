"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

def construct():
    deadline = time.time() + 115.0  # safety margin before 150s kill

    # Precompute small primes up to 2*10^6 for trial division
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    small_primes = sieve_primes(2 * 10**6)
    # Primes up to 1000 for quick residual checking in the window
    primes_up_to_1000 = [p for p in small_primes if p <= 1000]

    # Deterministic Miller-Rabin for n < 3.2e18
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def is_prime(n):
        """Fast prime check: small trial division then MR"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return False
        if n < 10**6:
            limit = int(math.isqrt(n))
            for sp in small_primes:
                if sp > limit:
                    break
                if n % sp == 0:
                    return False
            return True
        return is_prime_mr(n)

    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def is_p_smooth_small(n, p):
        """Check if all prime factors of n are <= p (for small p or general use)"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        for sp in small_primes:
            if sp * sp > n:
                break
            if sp > p:
                break
            if n % sp == 0:
                n //= sp
                if not is_p_smooth_small(n, p):
                    return False
                return True
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        d = pollard_brent(n)
        if d == n:
            return n <= p
        if not is_p_smooth_small(d, p):
            return False
        return is_p_smooth_small(n // d, p)

    def check_run_fast(p, max_k=200):
        """
        For p > 100: n = p^2 + i (1<=i<=200) has at most one prime factor > p.
        n is NOT p-smooth iff:
          (a) n is prime, OR
          (b) n = q * r where q is prime > p and r <= p
        We check by: divide n by small primes, then check if residual is prime.
        If residual > 1 and is_prime(residual) and residual > p: not p-smooth.
        If residual > 1 and composite: need further check (rare for p>100).
        """
        base = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = base + i
            if p <= 100:
                if not is_p_smooth_small(n, p):
                    break
            else:
                # Optimized check for p > 100
                # n < p^2 + 201, so sqrt(n) < p + 1
                # Any prime factor > p must be the "large" one
                # Divide by primes <= 1000 first (covers small factors)
                m = n
                for sp in primes_up_to_1000:
                    if sp * sp > m:
                        break
                    while m % sp == 0:
                        m //= sp
                # Now m is either 1, prime, or product of primes > 1000
                if m == 1:
                    pass  # p-smooth
                elif is_prime(m):
                    if m > p:
                        break  # not p-smooth
                    # m <= p, p-smooth
                else:
                    # m is composite with all factors > 1000
                    # For p > 100, m < (p^2+201)/1000 < p^2/100 for p > 14
                    # m could be product of two primes both > 1000
                    # Check if any factor > p
                    # Since m < p^2 + 201 and all prime factors > 1000:
                    # If m has a factor > p, the cofactor is < p (since m < p^2+201)
                    # So m = q * r with q > p prime, r < p+1
                    # Check: is m prime? Already checked no.
                    # Try to factor m
                    if m < 10**12:
                        # Use Pollard or trial division
                        d = pollard_brent(m)
                        if d == m:
                            # Shouldn't happen (is_prime said no), but safe
                            if m > p:
                                break
                        else:
                            if d > p or (m // d) > p:
                                # Check if the large factor is prime
                                large = max(d, m // d)
                                small = min(d, m // d)
                                if is_prime(large) and large > p:
                                    break
                                elif is_prime(small) and small > p:
                                    break
                                # Otherwise both factors <= p or not prime in problematic way
                                # Need deeper check
                                if not is_p_smooth_small(m, p):
                                    break
                    else:
                        if not is_p_smooth_small(m, p):
                            break
            k = i
        return k

    # Segmented sieve over [low, high]
    def segmented_sieve(low, high):
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        if limit <= 2 * 10**6:
            small = [p for p in small_primes if p <= limit]
        else:
            small = sieve_primes(limit)
        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes

    # Generate candidate primes and test them
    best_k = 0
    best_p = 2

    # 1. Test small primes first
    for p in small_primes:
        if p > 50000 and time.time() > deadline - 30:
            break
        k = check_run_fast(p)
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Search larger primes using segmented sieve
    # Focus on ranges where runs are likely to be longer
    current_low = 50000
    current_high = 100000

    while current_high <= 10**9 and time.time() < deadline - 10:
        candidates = segmented_sieve(current_low, current_high)

        for p in candidates:
            if time.time() > deadline - 5:
                break

            # Quick pre-check: test first few numbers to quickly reject
            base = p * p
            # Check p^2+1: if prime and > p, reject
            n1 = base + 1
            if is_prime(n1) and n1 > p:
                continue
            # Check p^2+2
            n2 = base + 2
            if is_prime(n2) and n2 > p:
                continue
            # Check p^2+3
            n3 = base + 3
            if is_prime(n3) and n3 > p:
                continue

            # If pre-check passes, do full check
            k = check_run_fast(p)
            if k > best_k:
                best_k = k
                best_p = p

        # Increase range
        current_low = current_high
        if current_high < 10**6:
            current_high = current_high * 2
        else:
            current_high = current_high + 200000

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END

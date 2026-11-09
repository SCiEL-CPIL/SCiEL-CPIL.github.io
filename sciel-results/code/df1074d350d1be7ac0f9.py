import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin

    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 2^64"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        """Pollard's rho with Brent's improvement"""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q_val = 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q_val = q_val * abs(x - y) % n
                    g = _gcd(q_val, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def sieve_primes(limit):
        """Sieve of Eratosthenes up to limit using bytearray for speed."""
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute small primes up to 10^6 for fast trial division
    SMALL_LIMIT = 10**6
    small_primes = sieve_primes(SMALL_LIMIT)

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        
        # Trial division by small primes
        # Limit division to min(p, sqrt(n)) but we only have primes up to SMALL_LIMIT
        # If p > SMALL_LIMIT, we stop at SMALL_LIMIT and handle remainder
        limit = min(p, int(n**0.5))
        if limit > SMALL_LIMIT:
            limit = SMALL_LIMIT
            
        for q in small_primes:
            if q > limit:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                    if n == 1:
                        return True
        
        if n == 1:
            return True
            
        # n is now prime or composite with all prime factors > SMALL_LIMIT
        # If p <= SMALL_LIMIT, then any remaining factor must be > p (since we divided all <= p)
        if p <= SMALL_LIMIT:
            return False

        # If p > SMALL_LIMIT:
        # Check if n is prime
        if is_prime_mr(n):
            return n <= p

        # n is composite.
        # Since n < (p+1)^2 approx, and we removed factors <= SMALL_LIMIT,
        # n must have a factor <= p (unless n is product of two primes > p, 
        # but then n > p^2, which is possible for small j? 
        # p^2+j < p^2+200. If p is large, p^2 is much larger than 200.
        # If n has two factors > p, n > p^2.
        # So n could be q1*q2 with q1,q2 > p.
        # In that case, n is not p-smooth.
        
        # Find a factor using Pollard-Brent
        f = pollard_brent(n)
        # If f > p, then n has a factor > p.
        # We need to check if f is p-smooth? No, if f > p, f itself might be composite.
        # But if f is a factor of n, and f > p, does it mean n is not p-smooth?
        # Not necessarily. f could be 2*p? No, 2 <= p.
        # If f > p, and f is prime, then n has prime factor > p. Fail.
        # If f is composite, it has prime factors. If all are <= p, then f is p-smooth.
        # But f > p.
        # So we recurse.
        return is_p_smooth(f, p) and is_p_smooth(n // f, p)

    best_k = 0
    best_p = 2

    # Search primes in increasing order
    # Start from 2
    p = 2
    while p <= 1000000000:
        if time.time() - start_time > DEADLINE:
            break
        
        if is_prime_mr(p):
            base = p * p
            k = 0
            while k < 200:
                if time.time() - start_time > DEADLINE:
                    break
                if is_p_smooth(base + k + 1, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
                if k >= 200:
                    break
        p += 1 if p == 2 else 2

    return {'p': best_p, 'k': best_k}

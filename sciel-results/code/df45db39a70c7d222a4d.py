import math
import random

def is_prime(n):
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Deterministic bases for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def pollard_brent(n):
    """Pollard's Brent's cycle finding algorithm for factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
        g, q, r = 1, 1, 1
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
                if g > 1:
                    break
        if g != n:
            return g

def factorize(n, small_primes):
    """Fully factor n into a multiset of prime factors, using small_primes for trial division."""
    if n == 1:
        return []
    factors = []
    # Trial division by small primes
    for p in small_primes:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
    else:
        # Use Pollard-Brent for the remaining part
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                stack.append(d)
                stack.append(m // d)
    return factors

def largest_prime_factor(n, small_primes):
    """Return the largest prime factor of n."""
    if n == 1:
        return 1
    factors = factorize(n, small_primes)
    return max(factors)

def construct():
    best_k = 0
    best_p = 2
    
    # Generate small primes up to 10^6 for fast trial division
    limit = 10**6
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Search over primes p
    # We can search a wider range now because trial division is fast for small factors.
    # Let's search p up to 10^5 first, then maybe jump to larger values if needed.
    # But the bottleneck is the "hard" factorization.
    # If p is small, p^2+i is small, so factorization is fast.
    # If p is large, p^2+i is large, but if it's p-smooth, it might have small factors.
    
    # Strategy:
    # 1. Iterate p from 2 to some limit (e.g., 10000).
    # 2. For each p, check k.
    
    search_limit = 10000
    for p in small_primes:
        if p > search_limit:
            break
        k = 0
        while k < 200:
            n = p * p + k + 1
            # Quick check: if n <= p, smooth (not possible here)
            # Check if n is p-smooth
            # We can optimize: if n has a prime factor > p, it fails.
            # largest_prime_factor does this.
            lpf = largest_prime_factor(n, small_primes)
            if lpf <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
            
    # If we didn't find a good k, we might need to search larger p.
    # But for now, let's return what we have.
    # To improve, we could random sample larger p.
    
    return {'p': best_p, 'k': best_k}

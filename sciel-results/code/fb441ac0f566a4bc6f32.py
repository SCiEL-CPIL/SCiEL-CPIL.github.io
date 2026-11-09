import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s kill
    
    def is_prime(n):
        """Deterministic Miller-Rabin for n < 3.3e24"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Witnesses sufficient for n < 3,317,044,064,679,887,385,961,981
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
    
    def pollard_brent(n):
        """Pollard's rho with Brent's improvement"""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def _factor(n, factors, p):
        """Recursively factor n, adding factors to list. Early termination if factor > p."""
        if n == 1:
            return True
        if is_prime(n):
            if n > p:
                return False
            factors.append(n)
            return True
        d = pollard_brent(n)
        if not _factor(d, factors, p):
            return False
        if not _factor(n // d, factors, p):
            return False
        return True
    
    def is_p_smooth(n, p, small_primes):
        """Check if all prime factors of n are <= p"""
        if n <= 1:
            return True
        
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            while n % q == 0:
                n //= q
            if n == 1:
                return True
        
        if n == 1:
            return True
        
        # n is now either 1, a prime, or a composite with factors > limit
        # If n is prime, check if n <= p
        if is_prime(n):
            return n <= p
        
        # n is composite, use Pollard's rho to factor it
        factors = []
        return _factor(n, factors, p)
    
    # Generate small primes for trial division
    def sieve(n):
        if n < 2:
            return []
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = 0
        return [i for i, v in enumerate(is_p) if v]
    
    # Precompute small primes up to 10000 for fast trial division
    small_primes = sieve(10000)
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit using sieve
    sieve_limit = 5_000_000
    primes = sieve(sieve_limit)
    
    # For each prime p, check consecutive integers p^2+1, ..., p^2+k
    # for p-smoothness. We want to maximize k.
    
    # Strategy: 
    # 1. Try all primes up to sieve_limit with fast pre-filter
    # 2. For each, compute how far the run goes
    # 3. Keep track of best
    
    for idx, p in enumerate(primes):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # Quick pre-filter: for very small p, the run will be very short
        if p < 100:
            continue
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() - start_time > DEADLINE:
                break
            val = p_sq + k + 1
            if is_p_smooth(val, p, small_primes):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            # If we hit 200, that's the max, we can stop
            if k >= 200:
                break
    
    # If time remains, try some larger primes
    if time.time() - start_time < DEADLINE - 30:
        # Try primes in ranges larger than sieve_limit
        # Sample random larger primes
        for _ in range(500):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random prime in [sieve_limit, 10^9]
            lo = max(sieve_limit + 1, 10**6)
            hi = min(10**9, 5*10**8)
            candidate = random.randrange(lo, hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            while not is_prime(candidate):
                candidate += 2
                if candidate > hi:
                    break
            
            if candidate > hi:
                continue
            
            k = 0
            p_sq = candidate * candidate
            while k < 200:
                if time.time() - start_time > DEADLINE - 1:
                    break
                val = p_sq + k + 1
                if is_p_smooth(val, candidate, small_primes):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}

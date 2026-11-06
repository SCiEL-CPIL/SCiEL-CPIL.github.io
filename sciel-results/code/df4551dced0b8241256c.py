import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145  # seconds, safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in small_primes:
            if a >= n:
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

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
                    if g != 1:
                        break
            if g != n:
                return g

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        if n <= p:
            return True
        
        # Trial division by small primes up to min(p, 1000)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for q in small_primes:
            if q > p:
                break
            while n % q == 0:
                n //= q
                if n <= p:
                    return True
        
        if n == 1:
            return True
        if n <= p:
            return True
        
        # If n is prime here, it must be > p
        if is_prime(n):
            return False
        
        # Find a factor using Pollard Rho
        d = pollard_rho(n)
        if d == n:
            # Should not happen for composite, but safety
            return is_prime(n) and n <= p
        
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def sieve_primes(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(limit + 1) if is_p[i]]

    best_k = 0
    best_p = 2

    # Check small primes first
    small_check_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    for p in small_check_primes:
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes
    # We want to find p such that p^2 + 1, ..., p^2 + k are all p-smooth.
    # This is equivalent to: for all i in 1..k, p^2 + i has no prime factor > p.
    # Or: for all prime q > p, p^2 not congruent to -i mod q for any i in 1..k.
    # This is hard to sieve directly for large p.
    # Instead, we search primes p and check the condition.
    
    # To speed up, we can pre-check: if p^2 + i has a small prime factor q > p, then it's not p-smooth.
    # But p^2 + i is roughly p^2, so its factors are roughly sqrt(p^2) = p.
    # So factors of p^2 + i are close to p.
    # If q is a prime factor of p^2 + i and q > p, then q is close to p (since q <= p^2+i).
    # Actually, if q > p and q divides p^2 + i, then p^2 ≡ -i (mod q).
    # So p is a square root of -i mod q.
    
    # Strategy: For a given p, check k from 1 to 200.
    # For each i, factor p^2 + i and check if all factors <= p.
    # To speed up factorization, we can first check if p^2 + i is divisible by any prime in (p, sqrt(p^2+i)].
    # But sqrt(p^2+i) is approx p. So we are looking for prime factors in (p, p^2+i].
    # Actually, if p^2 + i = q * m where q > p, then m < p^2/q < p.
    # So m < p.
    # So we can check: for each prime q > p, if q divides p^2 + i, then p^2 + i is not p-smooth.
    # But there are many primes q > p.
    
    # Alternative: Factor p^2 + i completely.
    # Since p^2 + i is about p^2, and we need to check if all factors <= p,
    # we can try to factor p^2 + i.
    # If p^2 + i is prime, then it's not p-smooth (since p^2 + i > p).
    # If p^2 + i = q * m with q > p, then not p-smooth.
    
    # Let's just use the is_p_smooth function.
    # To speed up, we can add a quick check: if is_prime(p^2 + i), return False immediately.
    
    def is_p_smooth_fast(n, p):
        if n <= p:
            return True
        if is_prime(n):
            return False
        return is_p_smooth(n, p)

    # Search primes up to 10^6 first, then 10^7, then 10^8, etc.
    limits = [10**6, 10**7, 10**8, 10**9]
    prev_limit = 200
    
    for limit in limits:
        if time.time() - start_time > deadline - 10:
            break
        
        # Sieve primes in (prev_limit, limit]
        primes = sieve_primes(limit)
        # Filter to (prev_limit, limit]
        primes = [p for p in primes if p > prev_limit]
        
        for p in primes:
            if time.time() - start_time > deadline - 5:
                break
            
            k = 0
            while k < 200:
                n = p * p + k + 1
                if is_p_smooth_fast(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
        
        prev_limit = limit

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)

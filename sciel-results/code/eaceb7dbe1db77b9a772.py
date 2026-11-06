import random
import time
import math
import sys

def construct():
    # Fast sieve for primes up to 10^6
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    best_k = 1
    best_s = 1
    
    # Phase 1: Exhaustive search for small numbers (fast)
    # We can go up to 10^5 or 10^6 quickly
    for s in range(1, 100000):
        seen = set()
        k = 0
        while k < 200:
            v = phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break

    # Phase 2: Structured Random Search
    # Generate candidates that are "prime-like" (prime or semi-prime with large factors)
    # to potentially avoid the dense low-phi collision zone.
    start_time = time.time()
    time_limit = 110
    
    # Helper to generate a random semi-prime p*q in range
    def random_semi_prime(low, high):
        # Pick p, q such that p*q in [low, high]
        # Let p be random prime in [2, sqrt(high)]
        # q = random prime in [low/p, high/p]
        # This is tricky. Simpler: pick random x in [low, high], factor it? No.
        # Let's just pick two random primes p, q and multiply, then check range.
        # Or pick p from primes, q from primes, if p*q in range, return.
        pass

    # Simpler: Just pick random s, but bias towards numbers with large prime factors?
    # We can't easily bias random integers.
    # Let's stick to random s but optimize the check.
    # Actually, let's try searching around primes.
    # If s is prime, phi(s) = s-1.
    # Let's pick random primes P in range, and test s = P, P-1, P+1, etc.
    
    # Generate random primes in range [10^6, 10^12]
    # We can't sieve up to 10^12.
    # We can pick random odd numbers and test primality?
    # Miller-Rabin is fast.
    
    def is_prime_mr(n, k=5):
        if n < 2: return False
        if n in (2, 3): return True
        if n % 2 == 0: return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def find_prime_near(n):
        # Find next prime >= n
        if n <= 2: return 2
        if n % 2 == 0: n += 1
        while True:
            if is_prime_mr(n):
                return n
            n += 2

    # Search around random primes
    while time.time() - start_time < time_limit:
        # Pick a random magnitude
        log_s = random.uniform(6, 12)
        target = int(10 ** log_s)
        
        # Find a prime near target
        p = find_prime_near(target)
        
        # Test a window around p
        # p is prime, so phi(p) = p-1.
        # We check s in [p-50, p+50]
        for s in range(max(1, p-50), min(10**12 - 200, p+50) + 1):
            seen = set()
            k = 0
            while k < 200:
                v = phi(s + k)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
                if best_k == 200:
                    break
        if best_k == 200:
            break

    return {'start': best_s, 'length': best_k}

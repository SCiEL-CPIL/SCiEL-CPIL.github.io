import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Small primes for quick trial division (first ~200 primes)
    small_primes = primes[:200]
    
    # Miller-Rabin deterministic for 64-bit integers
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    # Check if n is p-smooth
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Trial divide by small primes
        for q in small_primes:
            if q > p:
                break
            while n % q == 0:
                n //= q
            if n == 1:
                return True
        if n == 1:
            return True
        
        # If n is prime
        if is_prime_mr(n):
            return n <= p
        
        # If n is composite, find a factor
        # We need to check if ALL prime factors are <= p
        # If we find a factor d, we check d and n//d
        # But d might be composite.
        # We can recursively factor or just check if any prime factor > p
        
        def has_large_prime_factor(m, p):
            if m == 1:
                return False
            if is_prime_mr(m):
                return m > p
            d = pollard_rho(m)
            return has_large_prime_factor(d, p) or has_large_prime_factor(m // d, p)
            
        return not has_large_prime_factor(n, p)

    best_k = 0
    best_p = 2
    
    # Iterate through primes p
    # We can skip very small p as they are unlikely to have long runs
    for p in primes:
        if time.time() > deadline:
            break
        if p > 10**7: # Cap p to avoid too slow checks if we extend primes list
             break
             
        # Quick check for p^2+1
        # If p^2+1 is not p-smooth, skip
        if not is_p_smooth(p * p + 1, p):
            continue
            
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            n = p * p + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}

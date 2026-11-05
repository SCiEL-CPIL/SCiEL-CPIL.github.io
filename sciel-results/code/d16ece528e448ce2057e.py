import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Miller-Rabin
    # For n < 3,474,749,660,383, bases 2, 3, 5, 7, 11, 13, 17 are enough?
    # Actually for n < 3.2e9, bases 2, 3, 5, 7 are enough.
    # We are testing n up to ~10^18 (factors of p^2+i where p~10^7 -> p^2~10^14).
    # Deterministic MR for 64-bit integers requires bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Or the set {2, 325, 9375, 28178, 450775, 9780504, 1795265022} for < 2^64.
    
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
        
    # Bases for 64-bit determinism
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    
    for a in bases:
        if a % n == 0:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    while True:
        c = random.randrange(1, n)
        f = lambda x: (x * x + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    if n == 1:
        return []
    if is_prime(n):
        return [n]
    
    # Trial division for small primes to speed up
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return [p] + factorize(n // p)
            
    d = pollard_rho(n)
    if d == n:
        # Retry if pollard failed
        return factorize(n)
    return factorize(d) + factorize(n // d)

def sieve(limit):
    is_prime_arr = bytearray([1]) * (limit + 1)
    is_prime_arr[0] = 0
    is_prime_arr[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
    return [i for i, val in enumerate(is_prime_arr) if val]

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Generate primes up to a reasonable limit
    # Let's try 2,000,000 first. If fast, we can go higher, but we can't adapt easily in one shot.
    # 10^6 gave 2.333. Let's try 5 * 10^6.
    # Sieve of 5 * 10^6 is very fast (<0.1s).
    limit = 5 * 10**6
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    # We can also check small primes manually if they are not in the sieve or just use the list
    for p in primes:
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            n = p_sq + i
            # Factorize n
            # We need to check if all prime factors <= p
            # Optimization: If n is prime, its only factor is n. So if n > p, it fails.
            # If n is composite, we factor it.
            
            # Quick check: if n is prime
            if is_prime(n):
                if n > p:
                    break
                else:
                    k += 1
                    continue
            
            # If not prime, factorize
            factors = factorize(n)
            # Check if any factor > p
            # Since factors are primes, we just check max(factors)
            # But factorize returns a list.
            # We can optimize factorize to stop if a factor > p is found?
            # Standard factorize returns all.
            # Let's just check max.
            if max(factors) > p:
                break
            else:
                k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}

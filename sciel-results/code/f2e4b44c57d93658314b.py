import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    # Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
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
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    return g

def largest_prime_factor(n):
    if n <= 1:
        return 1
    factors = []
    while n > 1:
        if is_prime(n):
            factors.append(n)
            break
        d = pollard_brent(n)
        if d == n:
            # Fallback, shouldn't happen often
            d = 2
            while d * d <= n:
                if n % d == 0:
                    break
                d += 1
        factors.append(d)
        n //= d
    return max(factors)

def is_p_smooth(n, p):
    return largest_prime_factor(n) <= p

def sieve(n):
    is_prime_arr = bytearray([1]) * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i::i] = bytearray([0]) * len(is_prime_arr[i*i::i])
    return [i for i, v in enumerate(is_prime_arr) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate primes up to 10^7
    # 10^7 sieve takes ~0.1s and ~10MB memory
    limit = 10_000_000
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
        
        k = 0
        # Check up to 200
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
            
    return {'p': best_p, 'k': best_k}

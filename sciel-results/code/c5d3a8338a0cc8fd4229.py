import math
import time
import bisect

def is_prime_mr(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in witnesses:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = 2
        y = 2
        d = 1
        c = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
        c += 1

def factorize(n):
    if n == 1:
        return []
    if is_prime_mr(n):
        return [n]
    d = pollard_rho(n)
    return factorize(d) + factorize(n // d)

def sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def is_p_smooth_sieve(n, p, primes):
    v = n
    # Find index of p in primes
    idx = bisect.bisect_right(primes, p)
    for i in range(idx):
        q = primes[i]
        if q * q > v:
            break
        while v % q == 0:
            v //= q
    return v == 1

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_k = 0
    best_p = 2
    
    limit = 10**6
    primes = sieve(limit)
    
    for p in primes:
        if time.time() > deadline:
            break
            
        k = 0
        while k < 200:
            if time.time() > deadline:
                break
            n = p * p + k + 1
            if is_p_smooth_sieve(n, p, primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    if time.time() < deadline - 10:
        p = 10**6 + 1
        if p % 2 == 0:
            p += 1
        while p <= 10**9:
            if time.time() > deadline:
                break
            if is_prime_mr(p):
                k = 0
                while k < 200:
                    if time.time() > deadline:
                        break
                    n = p * p + k + 1
                    if is_p_smooth_large(n, p):
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
            p += 2
            
    return {'p': best_p, 'k': best_k}

def is_p_smooth_large(n, p):
    factors = factorize(n)
    for f in factors:
        if f > p:
            return False
    return True

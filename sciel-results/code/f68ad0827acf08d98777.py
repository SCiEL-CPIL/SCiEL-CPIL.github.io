import time
import random
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def is_prime(n):
        if n < 2: return False
        if n == 2 or n == 3: return True
        if n % 2 == 0 or n % 3 == 0: return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = random.randrange(1, n)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    def sieve(limit):
        if limit < 2: return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, isqrt(limit) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, limit + 1) if is_p[i]]

    small_primes = sieve(10000)
    
    def is_p_smooth(N, p):
        for q in small_primes:
            if q > N: break
            while N % q == 0:
                N //= q
        if N == 1: return True
        if is_prime(N):
            return N <= p
        d = pollard_brent(N)
        if d == N:
            return N <= p
        return is_p_smooth(d, p) and is_p_smooth(N // d, p)

    best_k = 0
    best_p = 2
    
    # Check primes in increasing order of magnitude
    # Start with small primes
    limits = [10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
    prev_limit = 2
    
    for limit in limits:
        if time.time() > deadline - 10:
            break
        primes = sieve(limit)
        for p in primes:
            if p <= prev_limit:
                continue
            if time.time() > deadline - 5:
                break
            k = 0
            for j in range(1, 201):
                if time.time() > deadline - 2:
                    break
                N = p * p + j
                if is_p_smooth(N, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 150:
                    return {'p': best_p, 'k': best_k}
        prev_limit = limit

    return {'p': best_p, 'k': best_k}

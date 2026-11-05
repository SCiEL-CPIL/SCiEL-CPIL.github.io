import math
import random
import time

def is_prime_mr(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
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
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
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

def largest_prime_factor(n, small_primes):
    if n < 2:
        return 1
    for p in small_primes:
        if p * p > n:
            break
        while n % p == 0:
            n //= p
    if n == 1:
        return 1
    if is_prime_mr(n):
        return n
    factors = []
    while not is_prime_mr(n):
        f = pollard_brent(n)
        while n % f == 0:
            factors.append(f)
            n //= f
    factors.append(n)
    return max(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    limit_small = 10**5
    is_prime_small = [True] * (limit_small + 1)
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if is_prime_small[i]:
            for j in range(i * i, limit_small + 1, i):
                is_prime_small[j] = False
    small_primes = [i for i in range(2, limit_small + 1) if is_prime_small[i]]
    
    best_k = 0
    best_p = 2
    
    def check_p(p):
        k = 0
        base = p * p
        while True:
            num = base + k + 1
            lpf = largest_prime_factor(num, small_primes)
            if lpf <= p:
                k += 1
                if k >= 200:
                    break
            else:
                break
        return k

    def sieve_segment(low, high):
        size = high - low + 1
        sieve = [True] * size
        limit = int(math.isqrt(high))
        for p in small_primes:
            if p > limit:
                break
            start = ((low + p - 1) // p) * p
            for j in range(start, high + 1, p):
                sieve[j - low] = False
        primes = []
        for i in range(size):
            if sieve[i] and (low + i) >= 2:
                primes.append(low + i)
        return primes

    # Search in specific windows
    windows = [
        (10**9 - 10**7, 10**9),
        (10**9 - 2 * 10**7, 10**9 - 10**7),
        (10**8, 10**8 + 10**7)
    ]
    
    for low, high in windows:
        if time.time() > deadline:
            break
        chunk_size = 10**5
        start = low
        while start < high and time.time() < deadline:
            end_chunk = min(start + chunk_size, high)
            primes = sieve_segment(start, end_chunk)
            for p in primes:
                if time.time() > deadline:
                    break
                k = check_p(p)
                if k > best_k:
                    best_k = k
                    best_p = p
            start = end_chunk + 1

    return {'p': best_p, 'k': best_k}

import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 120s limit

    # Sieve of Eratosthenes
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute primes up to 1,000,000 for trial division
    SIEVE_LIMIT = 1000000
    primes = sieve_primes(SIEVE_LIMIT)
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard Rho factorization
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x, y, d = 2, 2, 1
            c = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1
    
    # Check if n is p-smooth
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        m = n
        for q in primes:
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
                if m == 1:
                    return True
        if m == 1:
            return True
        if m <= p:
            return True
        if is_prime_mr(m):
            return False  # m is prime and > p
        d = pollard_rho(m)
        return is_p_smooth(d, p) and is_p_smooth(m // d, p)

    # Check run length for a given prime p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 2:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Search primes in increasing order
    for p in primes:
        if time.time() > DEADLINE - 5:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

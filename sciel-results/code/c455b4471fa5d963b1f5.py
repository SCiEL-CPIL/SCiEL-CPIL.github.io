import math
import time
import random
from collections import defaultdict

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Larger sieve for faster trial division
    SIEVE_LIMIT = 200000
    small_primes = sieve(SIEVE_LIMIT)
    small_primes_set = set(small_primes)
    
    # Cache for phi values
    phi_cache = {}
    
    def is_prime(n):
        if n < 2:
            return False
        if n in small_primes_set:
            return True
        if n < SIEVE_LIMIT * SIEVE_LIMIT:
            for p in small_primes:
                if p * p > n:
                    break
                if n % p == 0:
                    return False
            return True
        # For larger n, use Miller-Rabin
        def miller_rabin(n, a):
            if n % a == 0:
                return n == a
            d = n - 1
            r = 0
            while d % 2 == 0:
                d //= 2
                r += 1
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                return True
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    return True
            return False
        # Deterministic for n < 3.3e24
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if not miller_rabin(n, a):
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            c = random.randint(1, n - 1)
            x = random.randint(2, n - 1)
            y = x
            d = 1
            # Batch GCD for speed
            for _ in range(100):
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
                if d != 1:
                    break
            if d != n:
                return d

    def factor(n):
        if n in phi_cache:
            return None  # Already factored
        factors = []
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_prime(n):
                factors.append(n)
            else:
                def _factor_rec(m, out):
                    if m == 1:
                        return
                    if is_prime(m):
                        out.append(m)
                        return
                    d = pollard_rho(m)
                    _factor_rec(d, out)
                    _factor_rec(m // d, out)
                _factor_rec(n, factors)
        return factors

    def phi(n):
        if n == 1:
            return 1
        if n in phi_cache:
            return phi_cache[n]
        facs = factor(n)
        if facs is None:
            return phi_cache[n]
        unique_primes = set(facs)
        result = n
        for p in unique_primes:
            result -= result // p
        phi_cache[n] = result
        return result

    best_k = 1
    best_s = 1
    
    max_n = 10**12
    
    def check_window(s, max_len=200):
        seen = set()
        k = 0
        while k < max_len and s + k <= max_n:
            v = phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    random.seed(42)
    
    num_iterations = 0
    while time.time() - start_time < DEADLINE:
        s = random.randint(1, max_n - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
        
        if k >= 5:
            # More aggressive local search for good finds
            num_local = 20 if k >= 10 else 10
            for _ in range(num_local):
                if time.time() - start_time > DEADLINE:
                    break
                offset = random.randint(-5000, 5000)
                ns = s + offset
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns

        num_iterations += 1

    return {'start': best_s, 'length': best_k}

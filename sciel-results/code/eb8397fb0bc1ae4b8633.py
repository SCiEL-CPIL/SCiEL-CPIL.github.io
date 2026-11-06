import time
import random
import math
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # 1. Sieve of Eratosthenes for primes up to 10^6
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # 2. Precompute phi up to LIMIT
    phi = list(range(LIMIT + 1))
    for p in primes:
        for j in range(p, LIMIT + 1, p):
            phi[j] -= phi[j] // p
    
    # 3. Fast phi for n > LIMIT using Pollard's Rho
    def is_prime_large(n):
        if n <= LIMIT:
            return is_prime[n]
        # Miller-Rabin for n > LIMIT
        if n < 2:
            return False
        # Check small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Write n-1 as 2^r * d
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
        # For n < 10^12, these witnesses are sufficient
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
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
            c = random.randint(1, n - 1)
            x = random.randint(2, n - 1)
            y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factor(n, factors=None):
        if factors is None:
            factors = {}
        if n == 1:
            return factors
        if n <= LIMIT:
            # Use trial division with precomputed primes
            for p in primes:
                if p * p > n:
                    break
                while n % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    n //= p
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
            return factors
        # For n > LIMIT, use Pollard's Rho
        if is_prime_large(n):
            factors[n] = factors.get(n, 0) + 1
            return factors
        # Trial division by small primes first
        for p in primes[:100]:  # First 100 primes
            if p * p > n:
                break
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n //= p
        if n == 1:
            return factors
        if is_prime_large(n):
            factors[n] = factors.get(n, 0) + 1
            return factors
        # Use Pollard's Rho
        d = pollard_rho(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    def phi_large(n):
        if n <= LIMIT:
            return phi[n]
        facs = factor(n)
        result = n
        for p in facs:
            result -= result // p
        return result

    def get_phi(n):
        if n <= LIMIT:
            return phi[n]
        else:
            return phi_large(n)

    best_k = 1
    best_s = 1
    
    # 4. Scan small range
    # We can scan up to LIMIT - 200 using precomputed phi
    for s in range(1, LIMIT - 200):
        if time.time() - start_time > DEADLINE:
            break
        seen = set()
        k = 0
        # Max k is 200, but likely small.
        while k < 200:
            v = get_phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
    
    # 5. Random search for larger s with Pollard's Rho
    max_n = 10**12
    random.seed(42)
    
    def check_window(s):
        seen = set()
        k = 0
        while k < 200 and s + k <= max_n:
            v = get_phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Focus on ranges where phi values are more likely to be distinct
    # Numbers with many small prime factors have smaller phi values
    # Try to find windows where numbers have diverse factorizations
    
    while time.time() - start_time < DEADLINE:
        s = random.randint(LIMIT, max_n - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
    
    return {'start': best_s, 'length': best_k}

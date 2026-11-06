import time
import random
import math

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
        
    # 3. Fast phi for n > LIMIT using Pollard Rho
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

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
                d = gcd(abs(x - y), n)
            if d != n:
                return d

    def factor(n, factors=None):
        if factors is None:
            factors = {}
        if n == 1:
            return factors
        if n <= LIMIT:
            # Use precomputed? No, we need factors.
            # But we can just trial divide small n quickly.
            pass
        # Trial division by small primes first
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n //= p
        if n > 1:
            if n <= LIMIT:
                factors[n] = factors.get(n, 0) + 1
            else:
                # Use Pollard Rho for remaining part
                while n > 1:
                    if n <= LIMIT:
                        factors[n] = factors.get(n, 0) + 1
                        break
                    d = pollard_rho(n)
                    if d == n:
                        # Retry
                        continue
                    factors[d] = factors.get(d, 0) + 1
                    n //= d
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
    scan_limit = min(LIMIT, 10**7)
    
    for s in range(1, LIMIT - 200):
        if time.time() - start_time > DEADLINE - 10: # Leave time for random search
            break
        seen = set()
        k = 0
        while k < 200:
            v = get_phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            
    # 5. Random search for larger s
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

    # Use remaining time for random search
    while time.time() - start_time < DEADLINE:
        s = random.randint(LIMIT, max_n - 200)
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
            
    return {'start': best_s, 'length': best_k}

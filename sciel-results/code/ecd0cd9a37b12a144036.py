<<<<<<< SEARCH
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
        
        # 3. Fast phi for n > LIMIT
        # Use Pollard Rho or Trial Division with primes
        def factor(n, factors=None):
            if factors is None:
                factors = {}
            if n == 1:
                return factors
            if n <= LIMIT:
                # Use precomputed? No, factorization is needed for phi formula?
                # Actually, we have phi[n] directly.
                # But if we need factors for something else? No.
                pass
            # Trial division by small primes
            for p in primes:
                if p * p > n:
                    break
                while n % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    n //= p
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
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
=======
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
        
        # 3. Pollard's Rho for fast factorization
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
                x = random.randint(2, n-1)
                y = x
                c = random.randint(1, n-1)
                d = 1
                while d == 1:
                    x = (x*x + c) % n
                    y = (y*y + c) % n
                    y = (y*y + c) % n
                    d = gcd(abs(x - y), n)
                if d != n:
                    return d

        def factor(n, factors=None):
            if factors is None:
                factors = {}
            if n == 1:
                return factors
            # Trial division by small primes first
            for p in primes:
                if p * p > n:
                    break
                while n % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    n //= p
            if n > 1:
                # Use Pollard's Rho for remaining factor
                stack = [n]
                while stack:
                    m = stack.pop()
                    if m <= LIMIT:
                        # Factor using trial division
                        for p in primes:
                            if p * p > m:
                                break
                            while m % p == 0:
                                factors[p] = factors.get(p, 0) + 1
                                m //= p
                        if m > 1:
                            factors[m] = factors.get(m, 0) + 1
                    else:
                        d = pollard_rho(m)
                        if d == m:
                            # d is prime
                            factors[m] = factors.get(m, 0) + 1
                        else:
                            stack.append(d)
                            stack.append(m // d)
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
>>>>>>> REPLACE


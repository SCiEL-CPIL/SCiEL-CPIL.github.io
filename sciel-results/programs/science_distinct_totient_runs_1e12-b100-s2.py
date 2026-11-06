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

        best_k = 1
        best_s = 1
        
        # 4. Scan small range
        # We can scan up to LIMIT - 200 using precomputed phi
        scan_limit = min(LIMIT, 10**7) # If we extend limit?
        # Let's stick to LIMIT for precomp, but we can scan further if we compute on fly?
        # Scanning 10^6 with O(1) phi is very fast.
        
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
        
        # 5. Random search for larger s
        # If best_k is small, maybe larger s has larger k?
        # Unlikely, but let's try.
        
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

        while time.time() - start_time < DEADLINE:
            s = random.randint(LIMIT, max_n - 200)
            k = check_window(s)
            if k > best_k:
                best_k = k
                best_s = s
        
        return {'start': best_s, 'length': best_k}

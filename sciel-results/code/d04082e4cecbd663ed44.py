import time
import random
import math
import sys

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    max_range = 10**12
    best_k = 1
    best_s = 1
    
    # Miller-Rabin primality test
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Deterministic bases for n < 3,474,749,660,383
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's Rho factorization
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
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
    
    def factorize(n):
        """Factor n into prime factors with multiplicity"""
        factors = []
        if n <= 1:
            return factors
        # Try small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Use Pollard's Rho
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
            else:
                d = pollard_rho(m)
                stack.append(d)
                stack.append(m // d)
        return factors
    
    def phi(n):
        """Compute Euler's totient function"""
        if n <= 1:
            return n
        factors = factorize(n)
        result = n
        seen_primes = set()
        for p in factors:
            if p not in seen_primes:
                result -= result // p
                seen_primes.add(p)
        return result
    
    def check_run(start, max_len=200):
        """Check consecutive integers starting at 'start' for distinct totients"""
        seen = set()
        k = 0
        for i in range(max_len):
            val = phi(start + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    # Strategy 1: Exhaustive scan of small range
    # Scan from 1 up to some limit
    scan_limit = min(200000, max_range)
    i = 1
    while i <= scan_limit and time.time() < deadline - 5:
        k = check_run(i)
        if k > best_k:
            best_k = k
            best_s = i
            if best_k >= 200:
                break
        i += 1
    
    # Strategy 2: Targeted random search in larger ranges
    # Focus on regions where numbers are likely to have diverse factorizations
    attempts = 0
    while time.time() < deadline - 3:
        attempts += 1
        # Mix of strategies:
        # - Random points in [10^6, 10^12]
        # - Points near primorials or smooth numbers
        # - Points where consecutive numbers have different prime structures
        
        if attempts % 3 == 0:
            # Random in large range
            s = random.randint(10**6, max_range - 200)
        elif attempts % 3 == 1:
            # Try around multiples of small primorials
            # 2*3*5*7*11*13 = 30030
            # 2*3*5*7*11*13*17 = 510510
            # 2*3*5*7*11*13*17*19 = 9699690
            primorials = [30030, 510510, 9699690, 223092870]
            p = random.choice(primorials)
            multiplier = random.randint(1, max(1, max_range // p - 200))
            s = p * multiplier + random.randint(-100, 100)
            s = max(1, min(s, max_range - 200))
        else:
            # Random in medium range
            s = random.randint(10**5, 10**9)
        
        if time.time() > deadline - 2:
            break
            
        k = check_run(s)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
            
            # If we found a good run, try to extend it
            # Check left extension
            left_start = max(1, s - 50)
            for ls in range(s - 1, left_start - 1, -1):
                if time.time() > deadline - 2:
                    break
                # Check if phi(ls) is not in the set of phi values from s to s+best_k-1
                # We need to recompute or store, so let's just check a window
                test_k = check_run(ls)
                if test_k > best_k:
                    best_k = test_k
                    best_s = ls
                    if best_k >= 200:
                        break
                # If phi(ls) collides with any in the run, stop extending left
                # This is approximate but good enough for search
                
    return {'start': best_s, 'length': best_k}

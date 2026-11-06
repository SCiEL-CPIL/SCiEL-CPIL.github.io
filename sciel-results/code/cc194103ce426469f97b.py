"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
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
        factors = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors
    
    def phi_large(n):
        """Compute phi(n) for a single large number using Pollard's Rho."""
        if n == 1:
            return 1
        factors = factorize(n)
        result = n
        seen = set()
        for p in factors:
            if p not in seen:
                result -= result // p
                seen.add(p)
        return result
    
    # Find longest run of distinct totients starting at s, max length max_len
    def find_run_from(s, max_len=200):
        seen = set()
        k = 0
        for i in range(max_len):
            val = phi_large(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    # Strategy: Search in various ranges
    # For small ranges, we can be more exhaustive
    # For large ranges, we sample
    
    # First, check some small known good ranges
    small_starts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000, 1000000]
    for s in small_starts:
        if time_remaining() < 2:
            break
        k = find_run_from(s)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Search in larger ranges by sampling
    # We'll try different magnitudes
    search_bases = [
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11
    ]
    
    for base in search_bases:
        if time_remaining() < 5:
            break
        if base > 10**12 - 200:
            continue
        
        # Sample a few starting points in this range
        num_samples = 10
        for i in range(num_samples):
            if time_remaining() < 5:
                break
            s = base + random.randint(0, 1000)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = find_run_from(s)
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    MAX_N = 10**12
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute small primes for trial division
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    def is_prime(n):
        """Deterministic Miller-Rabin for n < 3.3e24."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Miller-Rabin with deterministic witnesses for 64-bit
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses sufficient for n < 3.3e24
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    
    def phi(x):
        """Compute Euler's totient function phi(x) exactly."""
        if x == 1:
            return 1
        result = x
        y = x
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def check_run(start, max_length, max_n):
        """Check consecutive integers from start for distinct phi values.
        Returns (length, start) of longest run found."""
        seen = set()
        k = 0
        for i in range(max_length):
            n = start + i
            if n > max_n:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k, start
    
    # Strategy: Search in different ranges to find long runs
    # Known results suggest runs of length ~10-20 are possible
    # Let's search systematically
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Dense search in small ranges where we can afford full phi computation
    # Search up to about 200,000 with step sizes
    
    # First, do a dense search in [1, 50000]
    search_limit = 50000
    for s in range(1, min(search_limit, int(MAX_N)) + 1):
        if time.time() > deadline - 5:
            break
        k, _ = check_run(s, 200, MAX_N)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Search in larger ranges with sparse sampling
    # Use random sampling in larger intervals
    if time.time() < deadline - 10:
        # Sample random starting points in various ranges
        ranges = [
            (10**5, 10**6),
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
            (10**9, 10**10),
            (10**10, 10**11),
            (10**11, 10**12),
        ]
        
        for lo, hi in ranges:
            if time.time() > deadline - 5:
                break
            num_samples = 200
            for _ in range(num_samples):
                if time.time() > deadline - 5:
                    break
                s = random.randint(lo, min(hi, int(MAX_N) - 200))
                k, _ = check_run(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 3: If we found a good run, try to extend the search around it
    if best_k > 5 and time.time() < deadline - 10:
        # Search around the best found start
        window = 1000
        for offset in range(-window, window + 1):
            if time.time() > deadline - 5:
                break
            s = best_s + offset
            if s < 1 or s + 199 > MAX_N:
                continue
            k, _ = check_run(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END

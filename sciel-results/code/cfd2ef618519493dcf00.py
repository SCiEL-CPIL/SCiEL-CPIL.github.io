import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s limit
    
    # 1. Precompute primes up to 10^6 for fast factorization
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def phi(x):
        """Compute Euler's totient function using precomputed primes."""
        if x <= limit:
            # Could use a sieve-based approach for small numbers, but trial division is fine
            pass
        
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result
    
    # 2. Search for long runs of consecutive integers with distinct totients
    # Strategy: Start from large numbers where collisions are rarer
    # Use a sliding window approach with early termination
    
    best_k = 1
    best_s = 1
    
    # First, do a quick scan of small numbers to find a decent baseline
    # Then, do random sampling at large scales
    
    # Quick scan of small range
    max_small = 10000
    phi_values = [0] + [phi(i) for i in range(1, max_small + 1)]
    
    for s in range(1, max_small - 200):
        if time.time() > deadline - 30:
            break
        seen = set()
        k = 0
        while s + k <= max_small and phi_values[s + k] not in seen:
            seen.add(phi_values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
    
    # Now do random sampling at large scales
    # Focus on numbers up to 10^12
    max_val = 10**12
    
    # We'll sample random starting points and check windows
    # To be efficient, we'll check windows of size up to 200
    
    num_attempts = 0
    max_attempts = 1000000  # Will be limited by time
    
    # Use a mix of random starts and starts near multiples of small primes
    # to avoid obvious collision patterns
    
    while time.time() < deadline - 10 and num_attempts < max_attempts:
        num_attempts += 1
        
        # Choose a random start point
        # Bias towards larger numbers where collisions are rarer
        # But also try some medium-sized numbers
        
        if num_attempts % 10 == 0:
            # Try medium numbers
            s = random.randint(10**6, 10**8)
        else:
            # Try large numbers
            s = random.randint(10**9, max_val - 200)
        
        # Check window of size up to 200
        seen = set()
        k = 0
        for i in range(200):
            n = s + i
            if n > max_val:
                break
            p = phi(n)
            if p in seen:
                break
            seen.add(p)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
    
    return {'start': best_s, 'length': best_k}

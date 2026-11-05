import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # Leave margin for overhead and evaluation setup
    
    # 1. Generate primes up to 10^6 using a fast bytearray sieve
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Efficient phi calculation using precomputed primes
    def phi(x):
        if x == 1:
            return 1
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

    # 3. Search for a run of length 200 with distinct totients
    best_k = 1
    best_s = 1
    
    # Helper to check a run starting at s
    def check_run(s, target_len=200):
        seen = set()
        k = 0
        for i in range(target_len):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    # First, try some deterministic starting points that are likely to work
    # Large even numbers or numbers near the upper bound are good candidates
    test_starts = [
        10**12 - 200,  # Near the upper bound
        10**11,
        10**10,
        10**9,
        5 * 10**11,
        3 * 10**11,
        7 * 10**10,
    ]
    
    for s in test_starts:
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        # Check time
        if time.time() - start_time > LIMIT:
            break

    # If deterministic didn't find 200, try random sampling
    random.seed(42)
    attempts = 0
    max_attempts = 1000
    
    while time.time() - start_time < LIMIT and attempts < max_attempts:
        s = random.randint(10**9, 10**12 - 199)
        
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
        
        attempts += 1
        
        # Periodic time check to avoid overshooting
        if attempts % 10 == 0:
            if time.time() - start_time > LIMIT:
                break

    return {'start': best_s, 'length': best_k}

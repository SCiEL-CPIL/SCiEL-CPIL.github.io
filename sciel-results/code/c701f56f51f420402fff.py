import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need to compute up to n+1 where n can be MAX_N
    
    # 1. Sieve of Eratosthenes to find primes up to LIMIT
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0].tolist()
    
    # 2. Initialize powerful_part array as float64 for vectorized multiplication
    powerful_part = np.ones(LIMIT + 1, dtype=np.float64)
    
    # 3. For each prime, process its powers using NumPy vectorized operations
    for p in primes:
        p_sq = p * p
        if p_sq > LIMIT:
            break
        
        p_pow = p_sq
        while p_pow <= LIMIT:
            next_pow = p_pow * p
            
            if next_pow > LIMIT:
                # All multiples of p_pow get p_pow multiplied
                idx = np.arange(p_pow, LIMIT + 1, p_pow)
                powerful_part[idx] *= p_pow
            else:
                # Multiples of p_pow that are NOT multiples of next_pow
                # k from 1 to max_k, skip k where k % p == 0
                max_k = LIMIT // p_pow
                k_vals = np.arange(1, max_k + 1)
                mask = (k_vals % p) != 0
                k_valid = k_vals[mask]
                idx = k_valid * p_pow
                powerful_part[idx] *= p_pow
            
            p_pow = next_pow
            
            # Deadline check
            if time.time() - start_time > deadline:
                break
    
    # Check deadline
    if time.time() - start_time > deadline:
        return MIN_N
    
    # 4. Scan all n in [MIN_N, MAX_N] and find max powerful_part[n] * powerful_part[n+1]
    # Convert to int for exact product computation
    pp = powerful_part.astype(np.int64)
    products = pp[MIN_N:MAX_N+1] * pp[MIN_N+1:MAX_N+2]
    best_idx = np.argmax(products)
    best_n = MIN_N + best_idx
    best_product = products[best_idx]
    
    # Safety check
    if best_n < MIN_N or best_n > MAX_N:
        best_n = MIN_N
    
    return int(best_n)

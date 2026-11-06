import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need P(n+1) for n=N, so up to N+1
    
    # 1. Generate primes up to sqrt(MAX_VAL) using simple sieve
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Initialize powerful part array
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # 3. Compute powerful parts using vectorized sieve
    for p in primes:
        p = int(p)
        q = p * p
        while q <= MAX_VAL:
            next_q = q * p
            
            if time.time() > deadline - 3:
                break
            
            # For multiples of q that are NOT multiples of next_q, multiply by q
            max_k = MAX_VAL // q
            if max_k > 0:
                # Create indices q, 2q, 3q, ..., max_k*q
                idx = np.arange(q, MAX_VAL + 1, q, dtype=np.int64)
                # k_vals = 1, 2, ..., max_k; keep those where k_vals % p != 0
                k_vals = idx // q
                mask = (k_vals % p) != 0
                if mask.any():
                    P[idx[mask]] *= q
            
            q = next_q
            
            if time.time() > deadline - 3:
                break
        
        if time.time() > deadline - 3:
            break
    
    # 4. Find best n in [10000, 1000000] using vectorized product
    products = P[10000:N+1] * P[10001:N+2]
    best_idx = int(np.argmax(products))
    best_n = 10000 + best_idx
    
    return best_n

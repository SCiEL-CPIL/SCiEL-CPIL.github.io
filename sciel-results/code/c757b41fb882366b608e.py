# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    N = 1000000
    MIN_N = 10000
    
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Sieve for Primes using NumPy
    is_prime = np.ones(N + 2, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    # Sieve of Eratosthenes
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0]

    # 2. Compute powerful part for all numbers up to N
    pp = np.ones(N + 2, dtype=np.int64)
    
    for p in primes:
        p_int = int(p)
        p2 = p_int * p_int
        if p2 > N:
            break
        
        power = p2
        while power <= N:
            next_power = power * p_int
            
            # Indices that are multiples of power
            idx = np.arange(power, N + 1, power)
            
            # Filter out multiples of next_power
            # We want idx where idx % next_power != 0
            mask = (idx % next_power) != 0
            idx_valid = idx[mask]
            
            if len(idx_valid) > 0:
                pp[idx_valid] *= power
            
            power = next_power
            
            if time.time() > deadline:
                return MIN_N

    # 3. Find max product
    # n in [MIN_N, N-1]
    # n+1 in [MIN_N+1, N]
    products = pp[MIN_N:N] * pp[MIN_N+1:N+1]
    
    best_idx = np.argmax(products)
    best_n = MIN_N + best_idx
    best_val = products[best_idx]

    return best_n

# EVOLVE-BLOCK-END

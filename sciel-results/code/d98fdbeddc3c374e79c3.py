import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 1000000
    min_n = 10000
    max_n = N  # n can be 1000000
    
    # We need pp up to N+1
    LIMIT = N + 1
    
    # Step 1: Generate primes up to sqrt(LIMIT) using Sieve of Eratosthenes
    limit = int(math.isqrt(LIMIT)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Step 2: Compute powerful part for all numbers up to LIMIT
    pp = [1] * (LIMIT + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        
        for m in range(p2, LIMIT + 1, p2):
            base_power = p2
            temp = m // p2
            power = base_power
            while temp % p == 0:
                temp //= p
                power *= p
            
            pp[m] *= power
    
    # Step 3: Scan for max product using NumPy
    # Convert to numpy array for fast vectorized multiplication
    pp_np = np.array(pp, dtype=np.int64)
    
    # We want n in [min_n, max_n]
    # Pairs: (pp[n], pp[n+1])
    # n ranges from min_n to max_n
    # So pp[n] ranges from pp[min_n] to pp[max_n]
    # pp[n+1] ranges from pp[min_n+1] to pp[max_n+1]
    
    left = pp_np[min_n:max_n + 1]
    right = pp_np[min_n + 1:max_n + 2]
    
    scores = left * right
    max_idx = np.argmax(scores)
    best_n = min_n + max_idx
    
    return best_n

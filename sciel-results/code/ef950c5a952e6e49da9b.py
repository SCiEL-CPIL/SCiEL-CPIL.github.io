import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need pp up to N+1
    
    # Array to store powerful parts, 1-indexed
    pp = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(max_val))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime, compute its contribution to powerful parts
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            continue
        
        # For each multiple m of p^2, find the exact power of p in m
        # m = k * p^2, where k ranges from 1 to max_val // p^2
        # The exponent of p in m is 2 + (exponent of p in k)
        
        max_k = max_val // p2
        for k in range(1, max_k + 1):
            m = k * p2
            # Find exact exponent of p in m
            # m is divisible by p^2, so let's find how many more p's divide k
            temp = k
            extra = 0
            while temp % p == 0:
                temp //= p
                extra += 1
            
            total_exp = 2 + extra
            pp[m] *= p ** total_exp
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Now find the max product pp[n] * pp[n+1] for n in [10000, 1000000]
    # Use numpy for vectorized computation
    products = pp[10000:N+1] * pp[10001:N+2]
    idx = np.argmax(products)
    best_n = 10000 + idx
    
    return int(best_n)

import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need pp up to N+1
    
    # 1. Sieve of Eratosthenes to get primes up to max_val
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Compute Q(n) = product of primes p where v_p(n) == 1
    Q = np.ones(max_val + 2, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        p2 = p * p
        
        if p2 <= max_val:
            # Mark multiples of p
            mask = np.zeros(max_val + 2, dtype=bool)
            mask[p::p] = True
            # Unmark multiples of p^2 (these have v_p >= 2)
            mask[p2::p2] = False
            # Multiply Q by p for remaining indices
            Q[mask] *= p
        else:
            # p^2 > max_val, so v_p(n) can be at most 1 for all n <= max_val
            # So all multiples of p have v_p = 1
            mask = np.zeros(max_val + 2, dtype=bool)
            mask[p::p] = True
            Q[mask] *= p
            
        # Check deadline
        if time.time() > deadline:
            break
    
    # 3. Compute Powerful Part P(n) = n // Q(n)
    # Note: n is divisible by Q(n)
    nums = np.arange(1, max_val + 2, dtype=np.int64)
    P = nums // Q
    
    # 4. Find max product P[n] * P[n+1] for n in [10000, 1000000]
    products = P[10000:N+1] * P[10001:N+2]
    idx = np.argmax(products)
    best_n = 10000 + idx
    
    return int(best_n)

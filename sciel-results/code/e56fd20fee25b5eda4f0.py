import math
import time
import numpy as np

DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    # Deadline guard: return best found if we exceed DEADLINE.
    # N is the upper bound for n.
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            # Optimization: start from i*i
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Initialize powerful part array
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts using NumPy vectorization
    for p in primes:
        power = p * p
        # We need power <= MAX_VAL
        while power <= MAX_VAL:
            next_power = power * p
            
            # Get indices divisible by 'power'
            indices = np.arange(power, MAX_VAL + 1, power)
            
            if next_power > MAX_VAL:
                # All these indices have exponent exactly e for prime p
                P[indices] *= power
                break
            
            # Filter indices not divisible by next_power
            mask = indices % next_power != 0
            filtered_indices = indices[mask]
            
            P[filtered_indices] *= power
            
            power = next_power
        
    # Check deadline before expensive scan
    if time.time() - start_time > DEADLINE:
        return 10000

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    # Use NumPy for vectorized product scan
    # products[i] = P[i] * P[i+1] for i in [10000, N]
    products = P[10000:N+1] * P[10001:N+2]
    idx = int(np.argmax(products))
    best_n = 10000 + idx

    return best_n

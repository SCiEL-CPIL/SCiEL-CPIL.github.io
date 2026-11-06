import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 1000000
    # We need P(n) for n up to 1000000 and P(n+1) up to 1000001
    # But n is in [10000, 1000000], so n+1 is in [10001, 1000001]
    # We need array size 1000002
    
    size = N + 2
    pow_part = np.ones(size, dtype=np.int64)
    
    # Generate primes up to sqrt(N)
    limit = int(math.isqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Sieve
    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            break
        # Iterate multiples of p2
        # Using a Python loop is fast enough for N=10^6
        for m in range(p2, size, p2):
            count = 0
            tmp = m
            while tmp % p == 0:
                tmp //= p
                count += 1
            pow_part[m] *= (p ** count)
            
    if time.time() - start_time > DEADLINE:
        return 10000

    # Find max product in range [10000, 1000000]
    # n from 10000 to 1000000
    # indices: 10000 to 1000000
    # products: pow_part[n] * pow_part[n+1]
    
    # Vectorized multiplication
    # Slice 1: 10000 to 1000000 (inclusive) -> length 990001
    # Slice 2: 10001 to 1000001 (inclusive) -> length 990001
    products = pow_part[10000:1000001] * pow_part[10001:1000002]
    
    best_idx = np.argmax(products)
    best_n = 10000 + best_idx
    best_val = products[best_idx]
    
    return int(best_n)

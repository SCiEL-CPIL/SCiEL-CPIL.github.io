import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need pp up to N+1
    
    # Array to store powerful parts, 1-indexed
    pp = [1] * (max_val + 1)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    
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
    best_n = 10000
    max_prod = 0
    
    for n in range(10000, N + 1):
        prod = pp[n] * pp[n + 1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
        
        # Check deadline periodically
        if (n - 10000) % 100000 == 0 and time.time() > deadline:
            break
    
    return best_n

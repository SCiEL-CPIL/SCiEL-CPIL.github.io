import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin
    
    N = 1000000
    MAX_VAL = N + 1  # We need P(n+1) for n=N, so up to N+1
    
    # 1. Generate primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # 2. Initialize powerful part array
    P = [1] * (MAX_VAL + 1)
    
    # 3. Compute powerful parts using sieve
    # For each prime p, for each power q = p^e (e >= 2):
    #   For each multiple m of q that is NOT a multiple of q*p:
    #     P[m] *= q
    # This correctly assigns the full p^e to m when the exponent of p in m is exactly e.
    
    for p in primes:
        q = p * p
        while q <= MAX_VAL:
            next_q = q * p
            
            if time.time() - start_time > deadline - 5:
                break
            
            # Iterate through multiples of q
            # m = k * q
            # We want to multiply P[m] by q if m is NOT a multiple of next_q
            # m is a multiple of next_q iff k is a multiple of p
            
            max_k = MAX_VAL // q
            
            # Process in blocks of p to skip multiples of p
            # For k in 1..max_k:
            #   if k % p != 0: P[k*q] *= q
            
            # More efficient: iterate k, skip when k % p == 0
            # We can do this by iterating over ranges that exclude multiples of p
            for k in range(1, max_k + 1):
                if k % p != 0:
                    P[k * q] *= q
                    
            q = next_q
            
        if time.time() - start_time > deadline - 5:
            break
    
    # 4. Find best n in [10000, 1000000]
    best_n = 10000
    best_val = P[10000] * P[10001]
    
    for n in range(10000, N + 1):
        val = P[n] * P[n+1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() - start_time > deadline - 2:
            break
            
    return best_n

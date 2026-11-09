import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80  # Safety margin before 90s limit
    
    MAX_N = 1000000
    MIN_N = 10000
    N = MAX_N + 1  # We need up to MAX_N (since we check n and n+1 for n up to MAX_N-1)
    
    # 1. Generate primes up to sqrt(N)
    limit = int(math.sqrt(N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # 2. Compute powerful part for all n in [1, N]
    # pp[n] will store the powerful part of n
    pp = [1] * (N + 1)
    
    # Optimization: Only compute for n >= MIN_N?
    # No, the sieve updates depend on multiples.
    # But we only CARE about n in [MIN_N, N].
    # However, the loops `for m in range(q, N+1, q)` are efficient.
    # We can restrict the range if we want, but q might be small.
    # It's faster to just do it for all N.
    
    for p in primes:
        q = p * p
        while q <= N:
            next_q = q * p
            # Iterate through multiples of q
            # If m is a multiple of q but not next_q, then v_p(m) is exactly the current power
            for m in range(q, N + 1, q):
                if m % next_q != 0:
                    pp[m] *= q
            q = next_q
    
    if time.time() - start_time > DEADLINE - 5:
        return MIN_N
        
    # 3. Find n in [MIN_N, MAX_N-1] that maximizes pp[n] * pp[n+1]
    best_product = -1
    best_n_final = MIN_N
    
    # Vectorized scan using numpy if available, else pure python
    # Pure python is fine for 10^6 iterations (~0.1s)
    for n in range(MIN_N, MAX_N):
        product = pp[n] * pp[n + 1]
        if product > best_product:
            best_product = product
            best_n_final = n
            
    return best_n_final

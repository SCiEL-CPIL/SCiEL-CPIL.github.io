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
    for p in primes:
        q = p * p
        while q <= MAX_VAL:
            next_q = q * p
            
            # Iterate through multiples of q
            # m = k * q
            # We want to multiply P[m] by q if m is NOT a multiple of next_q
            # m is a multiple of next_q iff k * q is a multiple of p * q iff k is a multiple of p
            # So if k % p != 0, then m is not a multiple of next_q
            
            # Optimization: iterate k from 1 to MAX_VAL // q
            # if k % p != 0: P[k*q] *= q
            
            # We can split this into blocks of p
            # For k in 1..MAX_VAL//q:
            #   if k % p != 0: ...
            
            # Let's just do the loop. It's efficient enough for small q.
            # For large q, the loop runs few times.
            
            if time.time() - start_time > deadline - 5:
                break
                
            max_k = MAX_VAL // q
            # Iterate k
            # To speed up, we can process in chunks or just simple loop
            for k in range(1, max_k + 1):
                if k % p != 0:
                    P[k * q] *= q
                    
            q = next_q
            
    # 4. Find best n
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

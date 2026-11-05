import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    LIMIT = N + 1  # We need powerful part of N+1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    # Initialize spf with itself
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Precompute powerful part for all numbers up to LIMIT
    # pp[i] = powerful part of i
    pp = [1] * (LIMIT + 1)
    pp[1] = 1
    
    for i in range(2, LIMIT + 1):
        if spf[i] == i:
            # Prime, powerful part is 1
            pp[i] = 1
        else:
            p = spf[i]
            j = i
            cnt = 0
            while j % p == 0:
                j //= p
                cnt += 1
            
            # j is i with all factors of p removed
            # pp[j] is already computed because j < i
            if cnt >= 2:
                pp[i] = (p ** cnt) * pp[j]
            else:
                pp[i] = pp[j]
                
    # Check time
    if time.time() > deadline:
        return MIN_N

    # 3. Find max product
    best_n = MIN_N
    best_val = 0
    
    # Local references for speed
    pp_local = pp
    min_n = MIN_N
    n_max = N
    
    for n in range(min_n, n_max + 1):
        val = pp_local[n] * pp_local[n+1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n

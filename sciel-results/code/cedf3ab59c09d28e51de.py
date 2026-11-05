import time
import math
import numpy as np

def construct():
    """
    Find n in [10000, 1000000] maximizing P(n) * P(n+1),
    where P(m) is the product of all prime powers p^e (e>=2) in the factorization of m.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5-second safety margin before 90s limit
    
    limit = 1000001  # We need P up to 1000001 (since n+1 can be 1000000)
    
    # Build smallest prime factor (SPF) sieve
    spf = np.arange(limit + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part P[m] for all m using the SPF table
    # P[m] = product of p^e for each prime p dividing m with exponent e >= 2
    P = np.ones(limit + 1, dtype=np.int64)
    
    for i in range(2, limit + 1):
        m = i
        res = 1
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        P[i] = res
    
    # Check deadline
    if time.time() > deadline:
        return 10000
    
    # Search for n in [10000, 999999] maximizing P[n] * P[n+1]
    best_n = 10000
    best_val = -1
    
    # Vectorized search in chunks to balance speed and deadline checking
    chunk_size = 50000
    for start in range(10000, 1000000, chunk_size):
        end = min(start + chunk_size, 1000000)
        # Compute products for n in [start, end)
        prods = P[start:end] * P[start + 1:end + 1]
        local_max_idx = np.argmax(prods)
        local_max_val = prods[local_max_idx]
        if local_max_val > best_val:
            best_val = local_max_val
            best_n = start + local_max_idx
        if time.time() > deadline:
            break
    
    return int(best_n)

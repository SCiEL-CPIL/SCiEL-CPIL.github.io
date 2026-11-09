import math
import time
import numpy as np

def _sigma(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _verify_solution(M, ks):
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        if k in seen:
            return False
        seen.add(k)
        if k < 1 or k * k > M:
            return False
        if k * _sigma(k) != M:
            return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    if not _verify_solution(best_M, best_ks):
        # Fallback to trivial
        best_M = 1
        best_ks = [1]
        best_count = 1

    # We will try to find better M using NumPy sieve.
    # We'll try a few N values.
    
    def try_sieve(N):
        nonlocal best_M, best_ks, best_count
        t0 = time.time()
        
        # 1. Sieve for sigma
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        # 2. Compute M = k * sigma(k)
        # k ranges from 1 to N
        ks_arr = np.arange(1, N + 1, dtype=np.int32)
        Ms_arr = ks_arr.astype(np.int64) * sigma[1:]
        
        # Free sigma to save memory
        del sigma
        
        # 3. Find duplicates
        # np.unique is O(N log N)
        unique_Ms, counts, inverse = np.unique(Ms_arr, return_counts=True, return_inverse=True)
        
        # 4. Find max count
        max_count = np.max(counts)
        
        if max_count > best_count:
            # Find indices of M with max count
            max_indices = np.where(counts == max_count)[0]
            
            # Try the first one
            # We might want to try a few to see if one is "better" (e.g. smaller M?)
            # But score is just count.
            idx = max_indices[0]
            M_val = int(unique_Ms[idx])
            
            # Get ks
            mask = (inverse == idx)
            ks_list = ks_arr[mask].tolist()
            
            # Verify
            if _verify_solution(M_val, ks_list):
                best_M = M_val
                best_ks = ks_list
                best_count = max_count
                
        return time.time() - t0

    # Try N=10^7
    N1 = 10_000_000
    try_sieve(N1)
    
    # If fast, try N=5*10^7
    if time.time() < DEADLINE - 20:
        N2 = 50_000_000
        try_sieve(N2)
        
    # If even faster, try N=10^8?
    # Memory might be high. 10^8 * 8 = 800MB for Ms.
    # Let's skip 10^8 to be safe.
    
    return {'M': int(best_M), 'k': [int(x) for x in best_ks]}

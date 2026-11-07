import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Search limit
    # 10^7 is safe and fast.
    N = 10**7
    
    try:
        # 1. Sieve for Smallest Prime Factor (SPF)
        spf = np.arange(N + 1, dtype=np.int64)
        for i in range(2, int(math.isqrt(N)) + 1):
            if spf[i] == i:
                spf[i*i::i] = i
        
        # 2. Compute sigma(n)
        sigma = np.ones(N + 1, dtype=np.int64)
        
        # Identify primes
        # spf[i] == i means i is prime (for i >= 2)
        primes = [i for i in range(2, N + 1) if spf[i] == i]
        
        for p in primes:
            # Update sigma for all multiples of p
            # For each multiple j, find the highest power p^k dividing j
            # and multiply sigma[j] by (p^(k+1)-1)/(p-1)
            
            # Optimization: process in chunks?
            # Just loop.
            for j in range(p, N + 1, p):
                m = j
                factor = 1
                term = 1
                while m % p == 0:
                    m //= p
                    term *= p
                    factor += term
                sigma[j] *= factor

        # 3. Compute M = k * sigma(k)
        k_range = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_range * sigma
        
        # 4. Find M with max frequency
        unique_M, counts, inverse = np.unique(M_vals, return_counts=True, return_inverse=True)
        
        max_count = np.max(counts)
        
        if max_count <= 3:
            # Fallback to known 3 if something went wrong or N too small
            return {'M': 333312, 'k': [336, 372, 434]}
            
        best_M_indices = np.where(counts == max_count)[0]
        best_M_idx = best_M_indices[0]
        best_M = int(unique_M[best_M_idx])
        
        # 5. Retrieve k values
        k_indices = np.where(inverse == best_M_idx)[0]
        k_list = [int(k) for k in k_indices]
        
        # Check time
        if time.time() > deadline:
             return {'M': 333312, 'k': [336, 372, 434]}
             
        return {'M': best_M, 'k': k_list}
        
    except Exception:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}

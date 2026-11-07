import math
import time
import numpy as np

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin

    def _fallback():
        # Known small example: M = 333312, k in {336, 372, 434}
        M = 333312
        ks = [336, 372, 434]
        ok = []
        for k in ks:
            if 1 <= k <= math.isqrt(M) and k * _sigma_slow(k) == M:
                ok.append(k)
        if len(ok) >= 1:
            return {'M': int(M), 'k': [int(x) for x in ok]}
        return {'M': 1, 'k': [1]}

    # Determine N based on available time and memory safety
    # N=50M int64 array is ~400MB. 
    # Sieve is O(N log N). 
    # Let's try N=50,000,000. If it's too slow, we'll hit the deadline.
    N = 50_000_000
    
    try:
        # Allocate sigma array
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Vectorized sigma sieve
        # This loop is the bottleneck. 
        # for d in 1..N: sigma[d::d] += d
        # Time complexity is roughly N * (1/1 + 1/2 + ... + 1/N) = N log N
        for d in range(1, N + 1):
            sigma[d::d] += d
            if time.time() > deadline - 20.0:
                break
                
    except Exception:
        return _fallback()

    if time.time() > deadline - 15.0:
        return _fallback()

    try:
        # Compute M_k = k * sigma(k)
        ks = np.arange(1, N + 1, dtype=np.int64)
        M_vals = ks * sigma[1:]  # sigma[1:] corresponds to k=1..N
        
        # Filter M <= 10^18
        # Note: 10^18 fits in int64 (max ~9.2*10^18)
        mask = M_vals <= 10**18
        M_valid = M_vals[mask]
        k_valid = ks[mask]
        
        if len(M_valid) == 0:
            return _fallback()
        
        # Find collisions
        # unique returns sorted unique values, inverse indices, and counts
        unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
        
        max_count = np.max(counts)
        
        if max_count < 2:
            return _fallback()
            
        # Find all M values that have the max count
        # There might be multiple M's with the same max count.
        # We pick the first one or the one with smallest M? 
        # The score is just the number of solutions.
        # Let's pick the M with the highest count. If tie, any is fine.
        # However, we must ensure k <= sqrt(M).
        # As derived, k <= sigma(k) is always true for k>1.
        # For k=1, M=1, 1<=1 is true.
        # So all k in the group are valid solutions.
        
        # Get the index of the first M with max count
        best_idx = np.argmax(counts)
        best_M = unique_M[best_idx]
        
        # Get all k values that map to best_M
        # We need to map back from M_valid to k_valid
        # inverse[i] is the index in unique_M for M_valid[i]
        target_indices = np.where(inverse == best_idx)[0]
        best_ks = k_valid[target_indices]
        
        if len(best_ks) < 2:
            return _fallback()
            
        # Convert to Python list of ints
        # Ensure they are distinct (they are, by construction from unique k range)
        # Sort them
        best_ks_sorted = sorted([int(k) for k in best_ks])
        
        # Final check: k <= isqrt(M)
        limit = math.isqrt(best_M)
        verified_ks = [k for k in best_ks_sorted if k <= limit]
        
        if not verified_ks:
            return _fallback()
            
        return {'M': int(best_M), 'k': verified_ks}
        
    except Exception:
        return _fallback()

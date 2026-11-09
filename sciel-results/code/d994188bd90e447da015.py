import sys
import math
import time
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin for 60s limit
    
    MAX_N = 10_000_000
    
    import numpy as np
    
    # 1. Compute sigma(n) for n in 1..MAX_N using divisor-sum sieve
    # sigma[n] = sum of all positive divisors of n
    # Sieve: for d in 1..N, add d to all multiples of d
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    for d in range(1, MAX_N + 1):
        if time.time() > deadline - 20:
            break
        sigma[d:MAX_N+1:d] += d
    
    if time.time() > deadline - 5:
        # Fallback: known valid pair with sigma(6)=12, sigma(11)=12
        return [11, 6]
    
    # 2. Get sigma values for n=1..MAX_N
    sigma_vals = sigma[1:MAX_N+1]  # shape (MAX_N,)
    
    # 3. Sort indices by sigma value to group equal sigma values
    sorted_idx = np.argsort(sigma_vals, kind='stable')
    
    # 4. Process groups of equal sigma values
    best_num = None  # numerator of error |2a-3b|
    best_den = None  # denominator of error 2b
    best_a = 1
    best_b = 1
    
    i = 0
    n = MAX_N
    while i < n:
        if time.time() > deadline - 3:
            break
        
        # Find the end of the group with the same sigma value
        j = i + 1
        while j < n and sigma_vals[sorted_idx[j]] == sigma_vals[sorted_idx[i]]:
            j += 1
        
        group_size = j - i
        if group_size >= 2:
            # Extract the n values in this group (1-indexed)
            group = 1 + sorted_idx[i:j]
            # Sort by n value for binary search
            group = np.sort(group)
            k = len(group)
            
            # For each b in group, find a in group minimizing |2a-3b|/(2b)
            for idx_b in range(k):
                b = int(group[idx_b])
                target = 1.5 * b
                
                # Binary search for a near 1.5*b
                pos = bisect.bisect_left(group, target)
                
                # Check a few positions around the insertion point
                for offset in range(-2, 3):
                    p = pos + offset
                    if 0 <= p < k:
                        a = int(group[p])
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue  # Error is exactly 0, invalid
                        den = 2 * b
                        
                        # Compare num/den with best_num/best_den using cross-multiplication
                        if best_num is None:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                        else:
                            if num * best_den < best_num * den:
                                best_num = num
                                best_den = den
                                best_a = a
                                best_b = b
        
        i = j
    
    if best_num is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]

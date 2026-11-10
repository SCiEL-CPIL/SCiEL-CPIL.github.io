import math
import numpy as np
from collections import defaultdict

def construct():
    # Approach: Sieve-based computation of sigma(k) for k up to N
    # Then group by M = k * sigma(k)
    
    # Choose N based on memory and time. 
    # N=10^7 takes some time but should be within 120s with NumPy.
    # Let's try N=20_000_000 if possible, or start with 10_000_000.
    N = 10_000_000
    
    # Initialize sigma array
    # Using int64 to hold sums. Max sigma(n) for n=10^7 is roughly n log log n ~ 10^7 * 3 = 3*10^7, fits in int64.
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Group k by M
    # We only care about M <= 10^18
    groups = defaultdict(list)
    
    # Iterate k
    # To speed up, we can iterate in chunks or just loop.
    # Looping 10^7 times in Python is slow (~1s per 10^6? No, simple loop is ~0.05s per 10^6? 
    # 10^7 iterations might take 0.5-1s. 
    # But we are accessing numpy array.
    
    for k in range(1, N + 1):
        s = sigma[k]
        M = k * s
        if M <= 10**18:
            groups[M].append(k)
            
    best_M = None
    best_ks = []
    
    for M, ks in groups.items():
        sqrt_M = math.isqrt(M)
        # Filter k <= sqrt_M
        # Since we iterated k in increasing order, ks is sorted.
        # We can use binary search to find the count.
        # But we need the list of ks for the final answer.
        # If len(ks) is large, we need to filter.
        
        # Optimization: only consider if len(ks) > len(best_ks)
        if len(ks) <= len(best_ks):
            continue
            
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_ks):
            best_ks = valid_ks
            best_M = M
            
    if best_M is None:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

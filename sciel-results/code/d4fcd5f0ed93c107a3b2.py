import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def run_search(N):
        # Use numpy for memory efficiency and speed
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve for sigma
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        # Compute M = k * sigma(k)
        # k is 1 to N
        ks = np.arange(1, N + 1, dtype=np.int64)
        M_vals = ks * sigma[1:] # sigma[1:] aligns with ks
        
        # Filter M <= 10^18 (though for N up to 10^7, M is small)
        # M_vals are int64, max val ~ 1.5e14, well within int64 range
        
        # Find most frequent M
        unique_M, inverse, counts = np.unique(M_vals, return_inverse=True, return_counts=True)
        
        max_count = np.max(counts)
        
        # Find indices of M with max count
        max_indices = np.where(counts == max_count)[0]
        
        if len(max_indices) == 0:
            return 0, []
            
        # Pick the first M with max count
        best_M_idx = max_indices[0]
        best_M = int(unique_M[best_M_idx])
        
        # Find all k's corresponding to this M
        k_indices = np.where(inverse == best_M_idx)[0]
        # k values are k_indices + 1
        best_ks = (k_indices + 1).tolist()
        
        return best_M, best_ks

    best_M = 0
    best_ks = []
    best_count = 0
    
    # Start with a smaller N to get a quick baseline
    N = 200000
    M, ks = run_search(N)
    if len(ks) > best_count:
        best_M, best_ks, best_count = M, ks, len(ks)
    
    # Try larger N values, checking time
    # The bottleneck is the sieve and unique. 
    # N=10^7 takes a few seconds. N=2*10^7 takes ~4x longer.
    
    target_Ns = [1000000, 5000000, 10000000, 20000000, 50000000]
    
    for N in target_Ns:
        # Estimate time: sieve is O(N log N) roughly, unique is O(N log N)
        # If we have less than 20s left, we might not finish a large N
        # A rough heuristic: time taken for N=10^7 is T. 
        # We need T * (N/10^7)^1.5 < remaining_time - 5
        # Let's just check remaining time and be conservative.
        
        remaining = deadline - time.time()
        
        # Heuristic: if N > 10^7, it's risky. 
        # Let's allow up to 50M if we have plenty of time.
        
        # More precise check:
        # Assume N=10^7 takes ~2-3s.
        # If remaining < 5s and N > 10^7, skip.
        # If remaining < 10s and N > 20M, skip.
        
        if N > 10000000 and remaining < 15:
            continue
        if N > 20000000 and remaining < 25:
            continue
        if N > 50000000 and remaining < 40:
            continue
            
        try:
            M, ks = run_search(N)
            if len(ks) > best_count:
                best_M, best_ks, best_count = M, ks, len(ks)
        except Exception:
            # If memory error or timeout-ish, break
            break
        if time.time() > deadline - 5:
            break

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin (evaluator kills at 150s)
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Candidate N values - larger N gives more chances for collisions
    N_candidates = [10**6, 2 * 10**6, 4 * 10**6, 8 * 10**6, 16 * 10**6, 30 * 10**6]
    
    for N in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        # Compute sigma using vectorized NumPy sieve
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break

        # Find M with most k using vectorized operations
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        
        # Filter M <= 10^18
        mask = M_arr <= 10**18
        M_valid = M_arr[mask]
        k_valid = k_arr[mask]
        
        if len(M_valid) == 0:
            continue
        
        # Find M with most k using np.unique
        unique_M, inverse = np.unique(M_valid, return_inverse=True)
        counts = np.bincount(inverse)
        max_count = int(counts.max())
        
        if max_count <= len(best_k):
            continue
        
        # Find all M that achieve max_count
        best_Ms = unique_M[counts == max_count]
        
        # Pick the first one (or we could pick any)
        current_best_M = int(best_Ms[0])
        
        # Get all k for this M
        current_best_k_list = [int(k) for k in k_valid[M_valid == current_best_M]]
        
        if len(current_best_k_list) > len(best_k):
            best_M = current_best_M
            best_k = current_best_k_list
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}

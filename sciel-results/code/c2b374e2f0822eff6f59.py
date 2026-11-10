import time
import math
import numpy as np

def construct():
    deadline = time.time() + 140
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    limits = [10**7, 2*10**7, 5*10**7]
    
    for N in limits:
        if time.time() > deadline - 40:
            break
            
        sigma = np.ones(N + 1, dtype=np.int64)
        
        for i in range(1, N + 1):
            sigma[i::i] += i
            
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:]
        
        mask = M_vals <= 10**18
        if not np.any(mask):
            continue
            
        M_filtered = M_vals[mask]
        k_filtered = k_vals[mask]
        
        if len(M_filtered) == 0:
            continue
            
        unique_M, counts, inverse = np.unique(M_filtered, return_counts=True, return_inverse=True)
        
        if len(unique_M) == 0:
            continue
            
        max_count = np.max(counts)
        
        if max_count > best_count:
            best_idx = np.argmax(counts)
            best_M_val = unique_M[best_idx]
            best_count = max_count
            
            best_k_vals = k_filtered[inverse == best_idx]
            
            best_M = int(best_M_val)
            best_k = [int(k) for k in best_k_vals]
            
            if best_count >= 100:
                break

    return {'M': int(best_M), 'k': [int(k) for k in best_k]}

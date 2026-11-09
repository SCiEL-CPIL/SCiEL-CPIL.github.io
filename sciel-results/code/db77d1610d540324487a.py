import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Known solution from seed
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy: Sieve up to N
    # 30M is a good balance of time and memory
    N = 30000000

    # If we are already close to deadline, skip sieve and return seed
    if time.time() > deadline - 20.0:
        return {'M': int(best_M), 'k': best_k}

    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This is the main bottleneck
    for i in range(1, N + 1):
        sigma_arr[i::i] += i
        if time.time() > deadline - 30.0:
            break

    # Compute M = k * sigma(k)
    k_arr = np.arange(1, N + 1, dtype=np.int64)
    M_arr = k_arr * sigma_arr
    
    # Find most frequent M
    unique_M, counts, inverse = np.unique(M_arr, return_counts=True, return_inverse=True)
    
    # Find max count
    max_count = np.max(counts)
    if max_count > best_count:
        best_count = int(max_count)
        best_M_idx = np.argmax(counts)
        best_M = int(unique_M[best_M_idx])
        
        # Find all k that map to this M
        mask = (inverse == best_M_idx)
        best_k = [int(x) for x in k_arr[mask]]
        
    return {'M': int(best_M), 'k': best_k}

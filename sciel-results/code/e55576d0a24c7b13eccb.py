import math
import numpy as np

def construct():
    # We'll search for M <= 10^18 with many k <= sqrt(M) such that k*sigma(k) = M
    # Strategy: Enumerate k up to N, compute k*sigma(k) using NumPy sieve, find mode.
    
    # Increase N to find denser collisions
    N = 20000000
    
    # Sieve for sigma function
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Compute M = k * sigma(k)
    ks = np.arange(1, N + 1, dtype=np.int64)
    vals = ks * sigma[1:]
    
    # Filter M <= 10^18
    mask = vals <= 10**18
    valid_ks = ks[mask]
    valid_vals = vals[mask]
    
    # Find the M with the highest frequency
    unique_vals, counts, inverse = np.unique(valid_vals, return_counts=True, return_inverse=True)
    
    max_count = np.max(counts)
    
    if max_count == 1:
        return {'M': 1, 'k': [1]}
        
    # Find an M with max_count
    best_M_idx = np.argmax(counts)
    best_M = int(unique_vals[best_M_idx])
    
    # Retrieve the k values for this M
    best_k_mask = (inverse == best_M_idx)
    best_ks = valid_ks[best_k_mask]
    
    best_k_list = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_k_list}

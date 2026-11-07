import math
import time
import numpy as np
from collections import defaultdict

def _sieve_sigma_numpy(limit):
    """Compute sigma(n) for all n <= limit using a NumPy sieve."""
    sigma = np.zeros(limit + 1, dtype=np.int32)
    for d in range(1, limit + 1):
        sigma[d::d] += d
    return sigma

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_M = None
    best_k_list = []
    
    # Try progressively larger limits
    limits = [1000000, 5000000, 20000000, 50000000]
    
    for limit in limits:
        if time.time() > deadline - 30.0:
            break
        
        t0 = time.time()
        sigma = _sieve_sigma_numpy(limit)
        t1 = time.time()
        
        # Compute M_vals = k * sigma(k) for k=1..limit
        ks = np.arange(1, limit + 1, dtype=np.int64)
        M_vals = ks * sigma[1:].astype(np.int64)
        
        # Filter M_vals <= 10^18 (should all be, but just in case)
        mask = M_vals <= 10**18
        M_vals = M_vals[mask]
        ks_valid = ks[mask]
        
        # Group by M_val
        # Use np.unique to find counts
        unique_M, counts, inverse = np.unique(M_vals, return_counts=True, return_inverse=True)
        
        # Find M with max count
        max_count = np.max(counts)
        if max_count > len(best_k_list):
            # Find all M with this count
            best_indices = np.where(counts == max_count)[0]
            # Pick the first one (or we could pick the one with smallest M)
            best_idx = best_indices[0]
            best_M_val = unique_M[best_idx]
            # Get all k values for this M
            best_ks = ks_valid[inverse == best_idx]
            best_k_list = [int(k) for k in best_ks]
            best_M = int(best_M_val)
        
        t2 = time.time()
        # print(f"Limit {limit}: sieve {t1-t0:.1f}s, group {t2-t1:.1f}s, best={len(best_k_list)}")
        
        # Early exit if we found a good one
        if len(best_k_list) >= 20:
            break
    
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    return {'M': best_M, 'k': best_k_list}

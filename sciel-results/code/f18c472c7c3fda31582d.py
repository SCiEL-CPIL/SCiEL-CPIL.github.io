import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate primes up to 10^6 (sqrt(10^12))
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # 2. Define segment size and search strategy
    # We will scan segments of size 5,000,000.
    seg_size = 5 * 10**6
    num_segments_to_try = 20 # Rough estimate of what fits in time
    
    for seg_idx in range(num_segments_to_try):
        if time.time() > deadline:
            break
            
        L = seg_idx * seg_size + 1
        R = L + seg_size - 1
        
        if R > 10**12:
            R = 10**12
            L = max(1, R - seg_size + 1)
            
        # Compute phi for [L, R]
        size = R - L + 1
        if size <= 0:
            continue
            
        # Initialize phi values
        phi_vals = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply sieve updates
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            first_mult = ((L + p - 1) // p) * p
            
            if first_mult > R:
                continue
                
            # Indices in phi_vals corresponding to multiples of p
            idx = np.arange(first_mult - L, size, p)
            
            # Update phi: phi[n] = phi[n] * (1 - 1/p) = phi[n] - phi[n]//p
            phi_vals[idx] -= phi_vals[idx] // p
        
        # Find longest run of distinct values in phi_vals
        sorted_indices = np.argsort(phi_vals, kind='stable')
        sorted_vals = phi_vals[sorted_indices]
        
        # Build last_occurrence array
        last_occurrence = np.full(size, -1, dtype=np.int64)
        
        if size > 1:
            value_changes = np.empty(size, dtype=bool)
            value_changes[0] = True
            value_changes[1:] = sorted_vals[1:] != sorted_vals[:-1]
            
            same_value_mask = ~value_changes
            same_value_mask[0] = False
            
            # Ensure we don't access out of bounds or use empty arrays
            if np.any(same_value_mask[1:]):
                np.put(last_occurrence, sorted_indices[1:][same_value_mask[1:]], 
                       sorted_indices[:-1][same_value_mask[1:]])
        
        # Compute longest run using sliding window
        left = 0
        max_k_seg = 0
        max_start_seg = 0
        
        # Optimized sliding window loop
        # We can break early if we find 200
        for right in range(size):
            prev = last_occurrence[right]
            if prev >= left:
                left = prev + 1
            
            cur_k = right - left + 1
            if cur_k > max_k_seg:
                max_k_seg = cur_k
                max_start_seg = left
                if max_k_seg == 200:
                    break
        
        # Update global best
        if max_k_seg > best_k:
            best_k = max_k_seg
            best_s = L + max_start_seg
            
            if best_k == 200:
                return {'start': int(best_s), 'length': int(best_k)}

    return {'start': int(best_s), 'length': int(best_k)}

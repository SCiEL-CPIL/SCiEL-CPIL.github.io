import time
import numpy as np

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N using a sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to all multiples of i
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 1000000 == 0 and time.time() - start_time > 35:
            break
    
    # Check deadline
    if time.time() - start_time > 38:
        return [int(11), int(6)]
    
    # Step 2: Group indices by sigma value
    indices = np.arange(1, N + 1)
    
    # Sort indices by their sigma value
    order = np.argsort(sigma[1:N+1])
    sorted_sigmas = sigma[1:N+1][order]
    sorted_indices = indices[order]
    
    # Find group boundaries where sigma value changes
    changes = np.diff(sorted_sigmas) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [N]))
    
    # Compute group sizes and sort by size descending (process largest groups first)
    group_sizes = np.diff(boundaries)
    group_order = np.argsort(-group_sizes)
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate over groups, largest first
    for gi in group_order:
        if time.time() - start_time > 55:
            break
        start = boundaries[gi]
        end = boundaries[gi+1]
        size = end - start
        
        if size <= 1:
            continue
            
        # Extract and sort group indices
        group_indices = np.sort(sorted_indices[start:end])
        
        # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
        # For each b, a should be close to 1.5 * b
        
        # Vectorized approach for the group
        targets = 1.5 * group_indices.astype(np.float64)
        pos = np.searchsorted(group_indices, targets, side='left')
        
        for k in range(size):
            b = int(group_indices[k])
            p = int(pos[k])
            
            # Check index p
            if p < size:
                a = int(group_indices[p])
                if a != b:
                    err = abs(2*a - 3*b) / (2*b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
            # Check index p-1
            if p > 0:
                a = int(group_indices[p-1])
                if a != b:
                    err = abs(2*a - 3*b) / (2*b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
    return [int(best_a), int(best_b)]

import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, 10s buffer)
    
    # Initialize with known seed
    best_M = 333312
    best_ks = [336, 372, 434]
    
    def update_best(M, ks):
        nonlocal best_M, best_ks
        # Convert numpy arrays to python lists/ints for return
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
        
        # Ensure all are Python ints
        ks_list = [int(k) for k in ks_list]
        
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list

    def search_with_numpy(K):
        local_start = time.time()
        
        try:
            # 1. Sieve for sigma using optimized approach
            # sigma[n] = sum of divisors of n
            # For each i, add i to all multiples of i
            sigma = np.zeros(K + 1, dtype=np.int64)
            
            # Optimize: process in batches and check time periodically
            batch_size = 10000
            for i in range(1, K + 1):
                # Check time every batch_size iterations
                if i % batch_size == 0 and time.time() > deadline - 5:
                    return None
                
                sigma[i::i] += i
            
            if time.time() > deadline - 5:
                return None
                
            # 2. Compute v(k) = k * sigma(k) for k = 1 to K
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]
            
            # 3. Filter valid M (M <= 10^18)
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return None
                
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            # 4. Sort and group by M value
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]
            
            # Find consecutive equal values
            if len(sorted_vals) > 1:
                diffs = np.diff(sorted_vals)
                change_points = np.where(diffs > 0)[0] + 1
            else:
                change_points = np.array([], dtype=np.int64)
            
            start_points = np.concatenate(([0], change_points))
            end_points = np.concatenate((change_points, [len(sorted_vals)]))
            
            counts = end_points - start_points
            
            if len(counts) == 0:
                return None
                
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            
            if max_count > 1:
                s = start_points[max_idx]
                e = end_points[max_idx]
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                update_best(M, ks_group)
                
            return 0 # Success
            
        except Exception as e:
            print(f"Error in search: {e}")
            return 1
            
    # Try multiple K values, starting with larger ones
    # K=50M uses ~400MB for sigma array, should be fine
    # If time permits, we can try multiple sizes
    target_Ks = [50000000, 20000000, 10000000, 5000000]
    
    for K in target_Ks:
        if time.time() > deadline - 30:
            break
            
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"Search K={K} finished with status {status}. Total elapsed: {elapsed:.2f}s")
        
        # If we got a valid result with a large K, we might still try smaller
        # ones to see if they find better solutions, but typically larger K is better
        # for finding more collisions. However, smaller K runs faster.
        # Let's continue if we have time.
        
    return {'M': best_M, 'k': best_ks}

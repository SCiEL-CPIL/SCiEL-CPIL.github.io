import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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
            
        # Ensure distinctness and validity
        ks_list = list(set(ks_list))
        ks_list.sort()
        
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list

    def search_with_numpy(K):
        local_start = time.time()
        
        try:
            # 1. Sieve for sigma
            # Use int32 to save memory. sigma(k) for k <= 2*10^8 is well within int32 range.
            # Max sigma(k) ~ k * log(log(k)) + ... 
            # For k=10^8, sigma(k) < 10^8 * 5 = 5*10^8 < 2^31.
            sigma = np.zeros(K + 1, dtype=np.int32)
            
            # Optimized sieve
            # The loop is the bottleneck.
            for i in range(1, K + 1):
                # Check time periodically
                if i % 100000 == 0 and time.time() > deadline - 15:
                    return 1
                
                # sigma[i::i] += i
                # This is a vectorized operation on a slice.
                sigma[i::i] += i
                
            if time.time() > deadline - 15:
                return 1
                
            # 2. Compute v(k) = k * sigma(k)
            ks = np.arange(1, K + 1, dtype=np.int64)
            # Cast sigma to int64 for multiplication to avoid overflow in intermediate steps if needed, 
            # but int32 * int64 -> int64 in numpy usually.
            vals = ks * sigma[1:K+1].astype(np.int64)
            
            # 3. Filter valid M
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return 1
                
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            # 4. Sort and group
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]
            
            # Find consecutive equal values
            diffs = np.diff(sorted_vals)
            change_points = np.where(diffs > 0)[0] + 1
            
            start_points = np.concatenate(([0], change_points))
            end_points = np.concatenate((change_points, [len(sorted_vals)]))
            
            counts = end_points - start_points
            
            if len(counts) == 0:
                return 1
                
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            
            if max_count > 1:
                s = start_points[max_idx]
                e = end_points[max_idx]
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                update_best(M, ks_group)
                
            return 0 # Success
            
        except MemoryError:
            print("Memory error, reducing K")
            return 1
        except Exception as e:
            print(f"Error in search: {e}")
            return 1
            
    # Try increasing K values
    # Start with a large K.
    # With int32, we can try larger K.
    # 10^8 takes ~400MB for sigma.
    # 2*10^8 takes ~800MB.
    # Let's try 10^8 first, then 2*10^8 if time permits.
    
    target_Ks = [100000000, 200000000]
    
    for K in target_Ks:
        if time.time() > deadline - 30:
            break
            
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"Search K={K} finished with status {status}. Total elapsed: {elapsed:.2f}s")
        
        # If we succeeded with a large K, we might not need smaller ones.
        # But if it timed out, we might have partial results? No, it returns early.
        # So we should try the largest feasible K first.
        
    return {'M': best_M, 'k': best_ks}

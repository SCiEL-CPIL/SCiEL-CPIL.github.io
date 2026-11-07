import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, leave 5s buffer
    
    # Initialize with known seed
    # M = 333312, k in [336, 372, 434] -> 3 solutions. Score 1.6667 implies 5 solutions found previously?
    # 5/3 = 1.666. So the current best is 5 solutions.
    # We need to find M with > 5 solutions.
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    def update_best(M, ks):
        nonlocal best_M, best_ks, best_count
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
        
        count = len(ks_list)
        if count > best_count:
            best_M = int(M)
            best_ks = ks_list
            best_count = count
            print(f"New best: M={M}, count={count}")

    def search_with_numpy(K):
        local_start = time.time()
        
        try:
            # 1. Sieve for sigma
            # Use int64 to avoid overflow, though k*sigma(k) can exceed int64?
            # k <= 5*10^7, sigma(k) ~ k log log k ~ 5*10^7 * 4 = 2*10^8.
            # k*sigma(k) ~ 10^16, which fits in int64 (9*10^18).
            sigma = np.zeros(K + 1, dtype=np.int64)
            
            # Optimization: Loop only up to K
            # To speed up, we can process in chunks or just rely on numpy speed
            # The python loop overhead is the main cost.
            
            # Check time periodically
            check_interval = max(10000, K // 200)
            
            for i in range(1, K + 1):
                if i % check_interval == 0:
                    if time.time() > deadline - 20:
                        return 1  # Time out
                
                sigma[i::i] += i
                
            if time.time() > deadline - 20:
                return 1
                
            # 2. Compute v(k) = k * sigma(k)
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]
            
            # 3. Filter valid M <= 10^18
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return 1
                
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            if len(vals) < best_count + 1:
                return 1
                
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
            
            if max_count > best_count:
                s = start_points[max_idx]
                e = end_points[max_idx]
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                update_best(M, ks_group)
                
            return 0  # Success
            
        except MemoryError:
            return 2  # Memory error
        except Exception as e:
            print(f"Error in search: {e}")
            return 3  # Other error

    # Try K values from largest to smallest
    # Estimate: K=10^7 takes ~10-20s. K=5*10^7 might take 50-100s.
    # Let's try 50M, 30M, 20M, 10M
    target_Ks = [50000000, 30000000, 20000000, 10000000]
    
    for K in target_Ks:
        if time.time() > deadline - 30:
            break
            
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"Search K={K} finished with status {status}. Total elapsed: {elapsed:.2f}s")
        
        if status == 0:
            # Success. We found the best in this range.
            # Smaller K will not find larger counts (since superset of k's).
            # So we can break.
            break
        elif status == 2:
            # Memory error, try smaller K
            continue
        else:
            # Timeout or error, try smaller K
            continue
    
    return {'M': best_M, 'k': best_ks}

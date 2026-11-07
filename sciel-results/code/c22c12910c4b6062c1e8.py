import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (150s limit)
    
    # Initialize with known seed
    best_M = 333312
    best_ks = [336, 372, 434]
    
    def update_best(M, ks):
        nonlocal best_M, best_ks
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
        ks_list = [int(k) for k in ks_list]
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list
    
    def search_with_numpy(K):
        try:
            # SPF sieve with vectorized inner loop
            spf = np.arange(K + 1, dtype=np.int64)
            spf[0] = 0
            spf[1] = 1
            for i in range(2, int(math.isqrt(K)) + 1):
                if time.time() > deadline - 10:
                    return None
                if spf[i] == i:
                    idx = np.arange(i * i, K + 1, i)
                    mask = spf[idx] == idx
                    spf[idx[mask]] = i
            
            # Compute sigma using multiplicative property (O(K))
            sigma = np.ones(K + 1, dtype=np.int64)
            pp = np.ones(K + 1, dtype=np.int64)
            rem = np.ones(K + 1, dtype=np.int64)
            
            for n in range(2, K + 1):
                if n % 2000000 == 0 and time.time() > deadline - 10:
                    return None
                p = spf[n]
                m = n // p
                if m % p != 0:
                    pp[n] = p
                    rem[n] = m
                    sigma[n] = sigma[m] * (1 + p)
                else:
                    pp[n] = pp[m] * p
                    rem[n] = rem[m]
                    sigma[n] = sigma[m] + pp[n] * sigma[rem[n]]
            
            if time.time() > deadline - 10:
                return None
                
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]
            
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return None
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]
            
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
            return 0
        except Exception as e:
            print(f"Error in search: {e}")
            return 1
    
    target_Ks = [50000000, 30000000, 15000000, 8000000]
    
    for K in target_Ks:
        if time.time() > deadline - 20:
            break
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"K={K}: status={status}, elapsed={elapsed:.1f}s, best={len(best_ks)}")
    
    return {'M': best_M, 'k': best_ks}

import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 150s limit, leave 10s buffer

    # Known seed: M = 333312 has solutions 336, 372, 434 (3 solutions)
    # Score 1.6667 implies 5 solutions found previously
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    def update_best(M, ks):
        nonlocal best_M, best_ks, best_count
        if isinstance(ks, (list, tuple)):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
        
        count = len(ks_list)
        if count > best_count:
            best_M = int(M)
            best_ks = ks_list
            best_count = count
            print(f"New best: M={M}, count={count}, elapsed={time.time()-start_time:.2f}s")

    def search_with_sieve(K):
        """Search for M with many k such that k*sigma(k) = M, using sieve up to K."""
        local_start = time.time()
        
        try:
            # Sieve for sigma(k)
            # sigma[k] = sum of divisors of k
            sigma = [0] * (K + 1)
            
            # Efficient sieve: for each i, add i to all multiples
            # This is O(K log K)
            for i in range(1, K + 1):
                for j in range(i, K + 1, i):
                    sigma[j] += i
            
            if time.time() > deadline - 15:
                return 1  # Time out
            
            # Compute v(k) = k * sigma(k) and group by value
            # Only consider k where k*sigma(k) <= 10^18
            max_M = 10**18
            groups = defaultdict(list)
            
            for k in range(1, K + 1):
                if k % 1000000 == 0 and time.time() > deadline - 15:
                    return 1
                v = k * sigma[k]
                if v <= max_M:
                    groups[v].append(k)
            
            if time.time() > deadline - 15:
                return 1
            
            # Find the group with the most elements
            max_count = 0
            max_M_val = None
            max_ks = None
            for M_val, ks_list in groups.items():
                cnt = len(ks_list)
                if cnt > max_count:
                    max_count = cnt
                    max_M_val = M_val
                    max_ks = ks_list
            
            if max_count > best_count and max_M_val is not None:
                update_best(max_M_val, max_ks)
            
            return 0  # Success
            
        except MemoryError:
            return 2  # Memory error
        except Exception as e:
            print(f"Error in search: {e}")
            return 3  # Other error

    def search_with_numpy(K):
        """Search using numpy for faster sigma computation."""
        local_start = time.time()
        
        try:
            import numpy as np
            
            # Sieve for sigma using numpy
            sigma = np.zeros(K + 1, dtype=np.int64)
            
            check_interval = max(10000, K // 200)
            
            for i in range(1, K + 1):
                if i % check_interval == 0:
                    if time.time() > deadline - 20:
                        return 1
                sigma[i::i] += i
            
            if time.time() > deadline - 20:
                return 1
                
            # Compute v(k) = k * sigma(k)
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]
            
            # Filter valid M <= 10^18
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return 1
                
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            if len(vals) < best_count + 1:
                return 1
                
            # Sort and group
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
            max_count = int(counts[max_idx])
            
            if max_count > best_count:
                s = int(start_points[max_idx])
                e = int(end_points[max_idx])
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                update_best(M, ks_group)
                
            return 0  # Success
            
        except MemoryError:
            return 2  # Memory error
        except Exception as e:
            print(f"Error in numpy search: {e}")
            return 3  # Other error

    # Try different K values
    # Start with larger K for more potential solutions, fall back to smaller if needed
    target_Ks = [20000000, 10000000, 5000000]
    
    for K in target_Ks:
        if time.time() > deadline - 30:
            break
        
        # Try numpy first (faster for large K)
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"NumPy search K={K} finished with status {status}. Total elapsed: {elapsed:.2f}s")
        
        if status == 0:
            # Success
            break
        elif status == 2:
            # Memory error, try smaller K
            continue
        else:
            # Timeout or error, try smaller K
            continue
    
    return {'M': best_M, 'k': best_ks}

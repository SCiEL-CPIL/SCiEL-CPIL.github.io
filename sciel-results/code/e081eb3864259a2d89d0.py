import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Best known fallback
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Try increasing limits
    # We need to be careful with memory.
    # N=10^7 -> 10M entries.
    # vals: 10M int64 -> 80MB
    # sigma: 10M int64 -> 80MB
    # k: 10M int64 -> 80MB
    # Total ~240MB. Should be fine.
    
    limits = [5_000_000, 10_000_000, 20_000_000, 50_000_000]
    
    for limit in limits:
        if time.time() > deadline - 20: # Leave 20s for processing
            break
            
        try:
            # Sieve sigma using NumPy
            sigma = np.zeros(limit + 1, dtype=np.int64)
            # This loop is the bottleneck
            # We can optimize by only iterating up to limit
            # But vectorized update is fast
            for i in range(1, limit + 1):
                sigma[i::i] += i
                
            k = np.arange(1, limit + 1, dtype=np.int64)
            vals = k * sigma
            
            # Filter vals <= 10^18
            # Since k <= 5e7, sigma ~ k log log k ~ 5e7 * 3 = 1.5e8
            # vals ~ 5e7 * 1.5e8 = 7.5e15 < 1e18
            # So no filtering needed for these limits
            # But for 50M, k=5e7, sigma ~ 2e8, vals ~ 1e16. Still < 1e18.
            
            # Sort by vals
            # We want to find duplicates
            # Sorting 10M elements takes time
            # np.argsort is O(N log N)
            
            if time.time() > deadline - 10:
                continue
                
            # We can use a hash map approach instead of sorting if we just want counts?
            # But we need the k's.
            # Sorting allows us to group easily.
            
            # To save time, we can just check if we can beat the current best (3)
            # If we find 4, we stop.
            
            # Let's use a dictionary for counts?
            # Iterating 10M items in Python is slow.
            # Vectorized sorting is better.
            
            # Get indices sorted by value
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            sorted_ks = k[sorted_idx]
            
            # Find groups of equal values
            # Compare adjacent elements
            # diff = np.diff(sorted_vals)
            # matches = diff == 0
            
            # This creates large arrays.
            # Alternative:
            # Loop through sorted_vals? No, too slow.
            
            # We can use np.unique with return_counts?
            # unique_vals, counts, inverse_indices = np.unique(sorted_vals, return_counts=True, return_inverse=True)
            # This is O(N log N) and memory heavy.
            
            # Let's try to find if any count >= 4.
            # If max count < 4, we don't care about the exact M unless it's 3 and we want to verify.
            # But we already have 3.
            
            # So we only care if max_count >= 4.
            
            # np.unique is implemented in C, so it's fast.
            unique_vals, counts, _ = np.unique(vals, return_counts=True)
            
            max_count = np.max(counts)
            
            if max_count > len(best_k):
                # Find the M with this count
                # There might be multiple
                # We just need one
                target_M = unique_vals[counts == max_count][0]
                
                # Get the k's for this M
                # mask = (vals == target_M)
                # found_ks = k[mask]
                
                # This mask operation is O(N)
                found_ks = k[vals == target_M]
                
                # Verify
                # Convert to Python ints
                found_ks_list = [int(x) for x in found_ks]
                
                # Check k <= sqrt(M)
                sqrt_M = math.isqrt(int(target_M))
                valid_ks = [x for x in found_ks_list if x <= sqrt_M]
                
                if len(valid_ks) > len(best_k):
                    best_M = int(target_M)
                    best_k = valid_ks
                    # If we found 4, we can return immediately?
                    # Or keep searching for 5?
                    # Unlikely to find 5.
                    # But let's check time.
                    if time.time() > deadline - 5:
                        break
            else:
                # If max_count == 3, we might find a different M with 3.
                # But we already have 3.
                # No improvement.
                pass
                
        except MemoryError:
            break
        except Exception as e:
            # print(e)
            break
            
    # Verify best_k with exact factorization
    def sigma_exact(n):
        if n <= 0:
            return 0
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                term = 1
                while temp % p == 0:
                    temp //= p
                    term = term * p + 1
                result *= term
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp + 1)
        return result

    verified = []
    for k in best_k:
        if k * sigma_exact(k) == best_M:
            verified.append(k)
            
    # Ensure distinct and sorted
    verified = list(set(verified))
    
    if len(verified) > 0:
        return {'M': int(best_M), 'k': [int(k) for k in verified]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}

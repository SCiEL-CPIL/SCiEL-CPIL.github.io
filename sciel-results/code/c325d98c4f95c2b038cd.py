import time
import numpy as np
from collections import defaultdict

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _sigma_sieve(n):
    """
    Compute sigma(k) for all k in [1, n] using a vectorized sieve.
    sigma(n) = sum of divisors.
    We can compute this by iterating i from 1 to n, and for each multiple j of i,
    adding i to sigma[j].
    This is O(n log n) but vectorized operations are fast.
    """
    sigma = np.zeros(n + 1, dtype=np.int64)
    # Initialize sigma[i] = i for all i? No, sigma is sum of divisors.
    # sigma[1] = 1.
    # For each i, add i to sigma[j] for all j multiple of i.
    for i in range(1, n + 1):
        # Add i to sigma[i], sigma[2i], sigma[3i], ...
        # We can do this with slicing: sigma[i::i] += i
        # However, this creates a temporary array or modifies in place.
        # NumPy slicing with step > 1 is fast.
        sigma[i::i] += i
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma for all n up to MAX_VAL
    # This might take a few seconds.
    sigma_arr = _sigma_sieve(MAX_VAL)
    
    # Check time
    if time.time() - start_time > TIME_LIMIT - 20:
        # Fallback if sieve is too slow (unlikely)
        return [11, 6] # sigma(11)=12, sigma(6)=12, |11/6 - 1.5| = |1.833 - 1.5| = 0.333
    
    # Build fibers: map from sigma value to list of indices
    # We only care about indices that have at least one other index with the same sigma.
    # To save memory and time, we can group them.
    # However, creating a defaultdict of lists for 10^7 items might be memory heavy.
    # Alternative: use np.unique to group.
    
    # Let's use a dictionary, but be careful.
    # Actually, iterating through 10^7 items in Python to build the dict is slow.
    # We can use numpy to sort by sigma value and then process groups.
    
    # Get indices sorted by sigma value
    # This is a stable sort, but we just need to group.
    # np.argsort is O(N log N).
    
    # To avoid high memory usage of argsort, we can do a bucket sort if sigma values are small,
    # but sigma(n) can be up to ~50*n, so ~5*10^8. Too large for buckets.
    
    # Let's try the argsort approach.
    try:
        # Sort indices by their sigma value
        # We want to group indices that have the same sigma.
        # sorted_indices = np.argsort(sigma_arr)
        # This creates an array of 10^7 integers (40 MB).
        # Then we can iterate through the sorted array and find boundaries where sigma changes.
        
        # To speed up, we can filter out indices where sigma is unique?
        # It's hard to know which are unique without checking.
        
        # Let's just do the sort.
        indices = np.arange(1, MAX_VAL + 1)
        sorted_indices = np.argsort(sigma_arr, kind='stable')
        sorted_sigma = sigma_arr[sorted_indices]
        
        # Now find groups where sorted_sigma[i] == sorted_sigma[i+1]
        # We can use np.diff to find where changes happen.
        # changes = np.diff(sorted_sigma) != 0
        # group_starts = np.concatenate(([0], np.where(changes)[0] + 1))
        # group_ends = np.concatenate((np.where(changes)[0] + 1, [len(sorted_sigma)]))
        
        # This might be memory intensive. Let's iterate through the sorted array in chunks or just loop.
        # A Python loop over 10^7 items is ~0.5s - 1s. Acceptable.
        
        best_pair = (11, 6) # Fallback
        best_num = abs(2*11 - 3*6) # 22-18=4
        best_den = 2*6 # 12
        best_err_val = best_num / best_den
        
        # We want to find pairs (a, b) in the same group minimizing |2a - 3b| / (2b)
        # Note: a and b are the values, not the indices in the sorted array.
        
        # Iterate through groups
        i = 0
        n_items = len(sorted_indices)
        
        # Optimization: We can skip groups of size 1.
        # We can also prioritize groups with larger values?
        # The error is |2a-3b|/(2b). If |2a-3b|=1, error is 1/(2b).
        # So larger b is better.
        # We should process groups that contain large numbers first?
        # Or just process all and keep the best.
        
        # To improve speed, we can check time periodically.
        
        while i < n_items:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Find the end of the current group
            j = i + 1
            current_sigma = sorted_sigma[i]
            while j < n_items and sorted_sigma[j] == current_sigma:
                j += 1
            
            group_size = j - i
            if group_size >= 2:
                # Extract the values in this group
                # vals = sorted_indices[i:j]
                # These vals are sorted because indices were sorted by sigma, 
                # but not necessarily by value.
                # We need to sort them by value to use two-pointer/binary search effectively?
                # Actually, if we sort the values in the group, we can do binary search for 1.5*b.
                
                vals = sorted_indices[i:j]
                # Sort vals by value
                vals = np.sort(vals)
                
                # For each b in vals, find a in vals closest to 1.5*b
                # Since vals is sorted, we can use np.searchsorted
                # targets = 1.5 * vals
                # indices_a = np.searchsorted(vals, targets)
                
                # Vectorized search:
                # For each b, we want a such that a is close to 1.5*b.
                # Let's compute candidates for all b at once.
                
                targets = 1.5 * vals
                # searchsorted returns the index where target would be inserted.
                # We need to check the element at that index and the previous one.
                
                idxs = np.searchsorted(vals, targets)
                
                # We need to check vals[idxs] and vals[idxs-1]
                # Handle boundaries: idxs can be 0 or len(vals)
                
                # Create candidate indices
                # For each b_idx, candidate a indices are idxs[b_idx] and idxs[b_idx]-1
                # We need to filter out invalid indices.
                
                cand_idx_1 = idxs
                cand_idx_0 = idxs - 1
                
                # Mask for valid indices
                valid_1 = cand_idx_1 < len(vals)
                valid_0 = cand_idx_0 >= 0
                
                # We will compute the error for these candidates.
                # a_1 = vals[cand_idx_1] where valid_1
                # a_0 = vals[cand_idx_0] where valid_0
                
                # To avoid complex masking in numpy, we can just iterate if the group is small,
                # or use numpy carefully.
                
                # Let's just iterate if group size is small, or use a simple loop for the group.
                # Most groups are small. Large groups are rare.
                if group_size < 100:
                    for b in vals:
                        # Binary search for 1.5*b in vals
                        pos = np.searchsorted(vals, 1.5 * b)
                        
                        # Check pos and pos-1
                        for a in [vals[pos] if pos < len(vals) else None, 
                                  vals[pos-1] if pos > 0 else None]:
                            if a is None:
                                continue
                            if a == b:
                                continue
                            if 2 * a == 3 * b:
                                continue
                            
                            err_num = abs(2 * a - 3 * b)
                            err_den = 2 * b
                            
                            # Compare err_num/err_den with best_num/best_den
                            # err_num * best_den < best_num * err_den
                            if err_num * best_den < best_num * err_den:
                                best_num = err_num
                                best_den = err_den
                                best_pair = (a, b)
                else:
                    # For large groups, use vectorized approach
                    # This is complex, so we might just skip or do a coarse check.
                    # Given time constraints, we might just iterate with time checks.
                    # But large groups are very rare.
                    for b in vals:
                        if time.time() - start_time > TIME_LIMIT - 5:
                            break
                        pos = np.searchsorted(vals, 1.5 * b)
                        
                        for a in [vals[pos] if pos < len(vals) else None, 
                                  vals[pos-1] if pos > 0 else None]:
                            if a is None:
                                continue
                            if a == b:
                                continue
                            if 2 * a == 3 * b:
                                continue
                            
                            err_num = abs(2 * a - 3 * b)
                            err_den = 2 * b
                            
                            if err_num * best_den < best_num * err_den:
                                best_num = err_num
                                best_den = err_den
                                best_pair = (a, b)
            
            i = j
            
    except Exception:
        # Fallback
        pass
    
    a, b = best_pair
    return [int(a), int(b)]

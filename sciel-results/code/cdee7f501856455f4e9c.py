import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_sigma(limit):
    """
    Precompute sigma(n) for all n <= limit using a sieve.
    Returns a numpy array of size limit+1 where index i holds sigma(i).
    """
    sigma = np.zeros(limit + 1, dtype=np.int64)
    for i in range(1, limit + 1):
        sigma[i::i] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit to be safe within 90s kill
    
    limit = 10**7
    
    # Precompute sigma
    sigma_arr = _precompute_sigma(limit)
    
    elapsed = time.time() - start_time
    if time.time() > deadline:
        return [11, 6]
    
    best_a, best_b = 11, 6
    best_num = 1  # |2*11 - 3*6| = |22-18| = 4... wait, let me recalc
    # sigma(11) = 1+11 = 12, sigma(6) = 1+2+3+6 = 12. Good.
    # |11/6 - 3/2| = |22/12 - 18/12| = 4/12 = 1/3
    # num = |2a - 3b| = |22 - 18| = 4, den = 2b = 12
    # So error = 4/12 = 1/3
    best_num = 4
    best_den = 12
    
    # Strategy: Search for pairs (a, b) such that sigma(a) == sigma(b)
    # and |a/b - 1.5| is minimized.
    # |2a - 3b| / (2b)
    
    # Vectorized targeted search for small k values
    # k = 2a - 3b, we want small |k|
    # For k=1: a = (3b+1)/2, b must be odd
    # For k=-1: a = (3b-1)/2, b must be odd
    # For k=3: a = (3b+3)/2 = 3(b+1)/2, b+1 even => b odd
    # For k=-3: a = (3b-3)/2 = 3(b-1)/2, b-1 even => b odd
    
    # Generate odd b's up to limit
    b_odd = np.arange(1, limit + 1, 2, dtype=np.int64)
    
    # Try k values in order of increasing |k|
    k_values = [1, -1, 3, -3, 5, -5, 7, -7, 9, -9, 11, -11, 13, -13, 15, -15]
    
    for k in k_values:
        if time.time() > deadline - 5:
            break
        
        try:
            # 2a = 3b + k => a = (3b + k) / 2
            # For a to be integer: 3b + k must be even
            # 3b + k even => b + k even (since 3b has same parity as b)
            # So b and k must have same parity.
            # k is odd in our list, so b must be odd. Good, b_odd are all odd.
            
            a_candidates = (3 * b_odd + k) // 2
            
            # Filter: 1 <= a <= limit
            mask = (a_candidates >= 1) & (a_candidates <= limit)
            b_valid = b_odd[mask]
            a_valid = a_candidates[mask]
            
            if len(b_valid) == 0:
                continue
            
            # Get sigma values
            sig_a = sigma_arr[a_valid]
            sig_b = sigma_arr[b_valid]
            
            # Find matches where sigma(a) == sigma(b)
            match_indices = np.where(sig_a == sig_b)[0]
            
            if len(match_indices) > 0:
                # We want to minimize |2a - 3b| / (2b) = |k| / (2b)
                # For fixed k, minimize 1/(2b) => maximize b
                # b_valid is sorted ascending (since b_odd is ascending and mask preserves order)
                # So the last match has the largest b
                idx = match_indices[-1]
                b_val = int(b_valid[idx])
                a_val = int(a_valid[idx])
                
                # Verify
                if 1 <= a_val <= limit and 1 <= b_val <= limit:
                    if int(sigma_arr[a_val]) == int(sigma_arr[b_val]):
                        num = abs(2 * a_val - 3 * b_val)
                        den = 2 * b_val
                        if num == 0:
                            continue
                        # Compare num/den < best_num/best_den
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a, best_b = a_val, b_val
                            # If k=1 or -1 with large b, this is likely very good
                            # We can break early if k is small and b is large
                            if abs(k) <= 1 and b_val > 100000:
                                return [int(best_a), int(best_b)]
                        
        except Exception as e:
            continue
    
    # If we found something better than fallback, return it
    if best_a != 11 or best_b != 6:
        return [int(best_a), int(best_b)]
    
    # Fallback: Full group-based search if time permits
    if time.time() < deadline - 15:
        try:
            # Group by sigma value
            sigma_arr_n = sigma_arr[1:]  # n=1..limit
            order = np.argsort(sigma_arr_n, kind='stable')
            sorted_sigma = sigma_arr_n[order]
            sorted_indices = order + 1  # n values (1-indexed)
            
            # Find group boundaries
            diff = np.diff(sorted_sigma)
            changes = np.where(diff != 0)[0] + 1
            boundaries = np.concatenate(([0], changes, [len(sorted_sigma)]))
            
            if time.time() > deadline - 5:
                return [int(best_a), int(best_b)]
            
            import bisect
            
            for i in range(len(boundaries) - 1):
                if time.time() > deadline - 3:
                    break
                
                lo = int(boundaries[i])
                hi = int(boundaries[i + 1])
                if hi - lo < 2:
                    continue
                
                group = sorted_indices[lo:hi]
                group_len = len(group)
                
                for b_idx in range(group_len):
                    b_val = int(group[b_idx])
                    target = (3 * b_val) // 2
                    
                    pos = bisect.bisect_left(group, target)
                    
                    lo_idx = max(0, pos - 2)
                    hi_idx = min(group_len, pos + 3)
                    
                    for idx in range(lo_idx, hi_idx):
                        a_val = int(group[idx])
                        if a_val == b_val:
                            continue
                        num = abs(2 * a_val - 3 * b_val)
                        if num == 0:
                            continue
                        den = 2 * b_val
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a, best_b = a_val, b_val
                            
        except Exception as e:
            pass
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END

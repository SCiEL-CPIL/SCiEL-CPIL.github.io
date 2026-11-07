"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys
from fractions import Fraction

MAX_N = 10_000_000

def construct():
    t0 = time.time()
    deadline = t0 + 55  # safety margin before 60s limit

    # Sieve for sigma(n)
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i

    # Check time
    if time.time() > deadline - 5:
        return [11, 6]

    # Group numbers by sigma value
    # We only need sigma values that appear at least twice
    # To save memory, use a dict mapping sigma -> list of numbers
    # But that's expensive. Instead, sort by sigma.
    
    # Create list of (sigma[n], n) pairs
    # This is 10M entries - use arrays
    import array
    # Actually, let's just sort indices by sigma value
    # sigma is a list of 10M+1 ints
    
    # Better approach: for each sigma value, collect all n with that sigma
    # Use a dictionary but only for sigma values that repeat
    # First pass: count occurrences
    # This is memory intensive. Let's use a different approach.
    
    # For each n, sigma[n] is known. We want to find pairs with same sigma.
    # Sort n by sigma[n], then adjacent elements in the sorted order have same sigma.
    
    # Create list of indices sorted by sigma value
    # 10M sort might be slow in pure Python...
    
    # Alternative: use numpy if available
    try:
        import numpy as np
        sigma_arr = np.array(sigma, dtype=np.int64)
        # Sort by sigma value
        order = np.argsort(sigma_arr[1:], kind='stable')
        sorted_sigma = sigma_arr[1:][order]
        sorted_n = order + 1
        
        # Find groups of same sigma
        # For each group, find pair closest to ratio 3/2
        best_err = float('inf')
        best_pair = (11, 6)
        
        i = 0
        n_total = len(sorted_n)
        while i < n_total:
            j = i + 1
            while j < n_total and sorted_sigma[j] == sorted_sigma[i]:
                j += 1
            group_size = j - i
            if group_size >= 2:
                # In this group, find a, b with same sigma, minimizing |a/b - 3/2|
                # a/b close to 3/2 means a ≈ 1.5b
                # For each b in group, ideal a = 1.5*b
                # Find a in group closest to 1.5*b
                # Group is sorted by n (since we sorted by sigma, and within same sigma, by n)
                # Actually sorted_n within same sigma group is sorted by n value
                
                group = sorted_n[i:j]
                # For each b in group, find a in group closest to 3*b/2
                # Use binary search
                for idx_b in range(len(group)):
                    b_val = int(group[idx_b])
                    target_a = 3 * b_val / 2  # a should be close to this
                    # Binary search in group for value closest to target_a
                    # group is sorted
                    lo, hi = 0, len(group) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if int(group[mid]) < target_a:
                            lo = mid + 1
                        else:
                            hi = mid
                    # Check lo and lo-1
                    candidates = []
                    if lo < len(group):
                        candidates.append(int(group[lo]))
                    if lo - 1 >= 0:
                        candidates.append(int(group[lo - 1]))
                    for a_val in candidates:
                        if a_val != b_val:
                            # Check sigma equality (should be guaranteed by grouping)
                            # Compute |a/b - 3/2|
                            # = |2a - 3b| / (2b)
                            err_num = abs(2 * a_val - 3 * b_val)
                            err_den = 2 * b_val
                            err = err_num / err_den
                            if err < best_err and err > 0:
                                best_err = err
                                best_pair = (a_val, b_val)
            i = j
            
            # Time check
            if time.time() > deadline - 3:
                break
        
        if best_pair != (11, 6):
            return [int(best_pair[0]), int(best_pair[1])]
        return [11, 6]
        
    except ImportError:
        # Pure Python fallback - will be slower
        # Sort by sigma
        pairs = [(sigma[n], n) for n in range(1, MAX_N + 1)]
        pairs.sort()
        
        best_err = float('inf')
        best_pair = (11, 6)
        
        i = 0
        n_total = len(pairs)
        while i < n_total:
            j = i + 1
            while j < n_total and pairs[j][0] == pairs[i][0]:
                j += 1
            group_size = j - i
            if group_size >= 2:
                group = [pairs[k][1] for k in range(i, j)]
                # group is sorted by n
                for idx_b in range(len(group)):
                    b_val = group[idx_b]
                    target_a = 3 * b_val / 2
                    # Binary search
                    lo, hi = 0, len(group) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if group[mid] < target_a:
                            lo = mid + 1
                        else:
                            hi = mid
                    candidates = []
                    if lo < len(group):
                        candidates.append(group[lo])
                    if lo - 1 >= 0:
                        candidates.append(group[lo - 1])
                    for a_val in candidates:
                        if a_val != b_val:
                            err_num = abs(2 * a_val - 3 * b_val)
                            err_den = 2 * b_val
                            err = err_num / err_den
                            if err < best_err and err > 0:
                                best_err = err
                                best_pair = (a_val, b_val)
            i = j
            if time.time() > deadline - 3:
                break
        
        if best_pair != (11, 6):
            return [int(best_pair[0]), int(best_pair[1])]
        return [11, 6]

# EVOLVE-BLOCK-END

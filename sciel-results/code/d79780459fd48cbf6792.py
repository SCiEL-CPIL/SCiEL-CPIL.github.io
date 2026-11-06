import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = 15
    
    # Generate all possible sums for first half (indices 0..14)
    sums_a = {0}
    for i in range(mid):
        w = W[i]
        new_sums = set()
        for s in sums_a:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_a = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Generate all possible sums for second half (indices 15..29)
    sums_b = {0}
    for i in range(mid, n):
        w = W[i]
        new_sums = set()
        for s in sums_b:
            new_sums.add(s + w)
            new_sums.add(s)
            new_sums.add(s - w)
        sums_b = new_sums
        
    if time.time() > deadline:
        return _fallback(W, n)
    
    # Convert to sorted arrays for vectorized binary search
    arr_b = np.array(sorted(sums_b), dtype=np.int64)
    b_min = arr_b[0]
    b_max = arr_b[-1]
    
    # Sort A sums by absolute value to find small |a+b| faster
    a_list = sorted(sums_a, key=lambda a: abs(a))
    arr_a = np.array(a_list, dtype=np.int64)
    
    # Vectorized binary search: find insertion points for -a in arr_b
    targets = -arr_a
    idx = np.searchsorted(arr_b, targets)
    
    # Compute candidate values from both sides of insertion point
    # idx[i] is the first index where arr_b >= targets[i]
    # So candidates are arr_b[idx[i]] and arr_b[idx[i]-1]
    
    best_val = None
    best_a = None
    best_b = None
    
    # Process in order of increasing |a| to find small |a+b| early
    for i in range(len(arr_a)):
        if time.time() > deadline:
            break
        
        a = int(arr_a[i])
        target = -a
        
        # Determine candidates based on whether target is within B range
        if target < b_min:
            # Best b is b_min
            b = b_min
            val = a + b
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b
                    if best_val == 1:
                        break
        elif target > b_max:
            # Best b is b_max
            b = b_max
            val = a + b
            if val != 0:
                abs_val = abs(val)
                if best_val is None or abs_val < best_val:
                    best_val = abs_val
                    best_a = a
                    best_b = b
                    if best_val == 1:
                        break
        else:
            # target is within B range, use precomputed idx
            idx_i = int(idx[i])
            candidates = []
            if idx_i < len(arr_b):
                candidates.append(int(arr_b[idx_i]))
            if idx_i > 0:
                candidates.append(int(arr_b[idx_i - 1]))
            
            for b in candidates:
                val = a + b
                if val != 0:
                    abs_val = abs(val)
                    if best_val is None or abs_val < best_val:
                        best_val = abs_val
                        best_a = a
                        best_b = b
                        if best_val == 1:
                            break
            if best_val == 1:
                break
    
    # If we found a solution, reconstruct coefficients
    if best_a is not None and best_b is not None:
        coeffs_a = _find_coeffs(best_a, 0, mid, W)
        coeffs_b = _find_coeffs(best_b, mid, n, W)
        
        if coeffs_a is not None and coeffs_b is not None:
            final_coeffs = coeffs_a + coeffs_b
            total = sum(c * W[i] for i, c in enumerate(final_coeffs))
            if total != 0:
                return final_coeffs
    
    return _fallback(W, n)

def _find_coeffs(target, start, end, W):
    """Recursively find coefficients for indices start to end (0-based, exclusive end)."""
    if start == end:
        return [] if target == 0 else None
        
    # Pruning: max possible absolute sum from start to end
    max_rem = sum(W[start:end])
    if abs(target) > max_rem:
        return None
        
    w = W[start]
    for c in [-1, 0, 1]:
        res = _find_coeffs(target - c * w, start + 1, end, W)
        if res is not None:
            return [c] + res
    return None

def _fallback(W, n):
    """Greedy fallback: construct a small nonzero sum."""
    coeffs = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        w = W[i]
        best_c = 0
        best_val = curr
        for c in [-1, 0, 1]:
            val = curr + c * w
            if abs(val) < abs(best_val):
                best_val = val
                best_c = c
        coeffs[i] = best_c
        curr = best_val
    
    if curr == 0:
        # Make nonzero by setting smallest weight to 1
        min_idx = min(range(n), key=lambda i: W[i])
        coeffs[min_idx] = 1
        curr = sum(c * W[i] for i, c in enumerate(coeffs))
    
    return coeffs

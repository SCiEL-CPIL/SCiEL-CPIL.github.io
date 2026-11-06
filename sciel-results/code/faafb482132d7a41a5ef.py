import math
import time
import numpy as np
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = 30
    H = reduce(math.lcm, range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = 15
    
    # Generate all possible sums for first half (indices 0..14)
    sums_a = [0]
    for i in range(mid):
        w = W[i]
        new_sums = []
        for s in sums_a:
            new_sums.append(s + w)
            new_sums.append(s)
            new_sums.append(s - w)
        # Deduplicate and sort
        sums_a = list(set(new_sums))
        if time.time() > deadline:
            return _fallback(W, n)
    
    # Generate all possible sums for second half (indices 15..29)
    sums_b = [0]
    for i in range(mid, n):
        w = W[i]
        new_sums = []
        for s in sums_b:
            new_sums.append(s + w)
            new_sums.append(s)
            new_sums.append(s - w)
        # Deduplicate and sort
        sums_b = list(set(new_sums))
        if time.time() > deadline:
            return _fallback(W, n)
    
    # Sort both for efficient search
    sums_a.sort()
    sums_b.sort()
    
    arr_b = np.array(sums_b, dtype=np.int64)
    
    best_val = None
    best_a = None
    best_b = None
    
    # Use two-pointer technique: for each a in sorted order, find best b
    # Since we want |a + b| minimized, for sorted a and b, we can use
    # a two-pointer approach
    # Actually, let's iterate over sorted a_list and use binary search on b
    # Sort a by absolute value to find small values first
    a_sorted = sorted(sums_a, key=abs)
    
    for a in a_sorted:
        if time.time() > deadline:
            break
        
        target = -a
        idx = np.searchsorted(arr_b, target)
        
        # Check neighbors
        for j in [idx - 1, idx]:
            if 0 <= j < len(arr_b):
                b = int(arr_b[j])
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
                # Local search polish: try flipping individual coefficients
                final_coeffs = _local_search_polish(final_coeffs, W, n)
                return final_coeffs
    
    return _fallback(W, n)

def _local_search_polish(coeffs, W, n):
    """Try to reduce |sum| by flipping individual coefficients."""
    curr = sum(c * W[i] for i, c in enumerate(coeffs))
    if curr == 0:
        return coeffs
    
    improved = True
    while improved:
        improved = False
        curr_abs = abs(curr)
        for i in range(n):
            w = W[i]
            c_old = coeffs[i]
            # Try changing c_old to c_new
            for c_new in [-1, 0, 1]:
                if c_new == c_old:
                    continue
                new_curr = curr + (c_new - c_old) * w
                if new_curr == 0:
                    continue
                if abs(new_curr) < curr_abs:
                    coeffs[i] = c_new
                    curr = new_curr
                    curr_abs = abs(curr)
                    improved = True
                    break
    return coeffs

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
    """Greedy fallback with local search: construct a small nonzero sum."""
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
    
    # Local search polish
    coeffs = _local_search_polish(coeffs, W, n)
    return coeffs

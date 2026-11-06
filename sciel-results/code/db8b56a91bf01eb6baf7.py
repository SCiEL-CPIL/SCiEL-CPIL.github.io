import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    weights = [L // (i + 1) for i in range(n)]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    # Generate sums
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    N2 = len(sums2_sorted)
    
    # Vectorized search for best pair
    # We want to minimize |s1 + s2| such that s1 + s2 != 0
    
    # For each s1 in sums1, the best s2 is near -s1
    # Use searchsorted to find insertion point of -s1 in sums2_sorted
    
    targets = -sums1
    # searchsorted returns index where target would be inserted to maintain order
    # idx is the first index such that sums2_sorted[idx] >= target
    # Candidates are idx-1, idx, idx+1
    
    idx = np.searchsorted(sums2_sorted, targets)
    
    # Clip indices to valid range
    idx_c0 = np.clip(idx - 1, 0, N2 - 1)
    idx_c1 = np.clip(idx, 0, N2 - 1)
    idx_c2 = np.clip(idx + 1, 0, N2 - 1)
    
    k2_c0 = sums2_sorted[idx_c0]
    k2_c1 = sums2_sorted[idx_c1]
    k2_c2 = sums2_sorted[idx_c2]
    
    sum_c0 = sums1 + k2_c0
    sum_c1 = sums1 + k2_c1
    sum_c2 = sums1 + k2_c2
    
    abs_c0 = np.abs(sum_c0)
    abs_c1 = np.abs(sum_c1)
    abs_c2 = np.abs(sum_c2)
    
    # Exclude zero
    INF = np.iinfo(np.int64).max
    abs_c0[sum_c0 == 0] = INF
    abs_c1[sum_c1 == 0] = INF
    abs_c2[sum_c2 == 0] = INF
    
    # Find global minimum
    min0 = np.min(abs_c0)
    min1 = np.min(abs_c1)
    min2 = np.min(abs_c2)
    
    best_abs = min(min0, min1, min2)
    
    if best_abs == INF:
        # Fallback, should not happen
        return [0]*30

    # Identify the best s1 and s2
    if best_abs == min0:
        idx_best = np.argmin(abs_c0)
        k1_best = sums1[idx_best]
        k2_best = k2_c0[idx_best]
    elif best_abs == min1:
        idx_best = np.argmin(abs_c1)
        k1_best = sums1[idx_best]
        k2_best = k2_c1[idx_best]
    else:
        idx_best = np.argmin(abs_c2)
        k1_best = sums1[idx_best]
        k2_best = k2_c2[idx_best]
        
    # Reconstruct coefficients using DFS with pruning
    def find_coeffs(weights, target):
        n_w = len(weights)
        # Weights are already in descending order of magnitude (L/1, L/2, ...)
        # Precompute suffix sums for pruning
        suffix_sums = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
        coeffs = [0] * n_w
        
        def dfs(idx, current_sum):
            if idx == n_w:
                return current_sum == target
            
            # Pruning: if the difference between target and current_sum
            # is larger than the sum of absolute values of remaining weights,
            # we can't reach the target.
            if abs(target - current_sum) > suffix_sums[idx]:
                return False
            
            w = weights[idx]
            
            # Try adding 1
            coeffs[idx] = 1
            if dfs(idx + 1, current_sum + w):
                return True
            
            # Try adding 0
            coeffs[idx] = 0
            if dfs(idx + 1, current_sum):
                return True
            
            # Try subtracting 1
            coeffs[idx] = -1
            if dfs(idx + 1, current_sum - w):
                return True
            
            return False
        
        if dfs(0, 0):
            return coeffs
        return None
        
    c1 = find_coeffs(w1, k1_best)
    c2 = find_coeffs(w2, k2_best)
    
    if c1 is None or c2 is None:
        # This should not happen if MITM found a valid sum
        return [0]*30
        
    return c1 + c2

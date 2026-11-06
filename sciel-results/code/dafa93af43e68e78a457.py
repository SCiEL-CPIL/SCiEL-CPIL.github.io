import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    weights = [L // (i + 1) for i in range(n)]
    
    # Sort weights in descending order for better DFS pruning
    # But we need to track original positions
    indexed_weights = [(w, i) for i, w in enumerate(weights)]
    indexed_weights.sort(key=lambda x: -x[0])
    
    # Split into two halves
    mid = n // 2
    w1_idx = indexed_weights[:mid]
    w2_idx = indexed_weights[mid:]
    
    w1 = [w for w, i in w1_idx]
    w2 = [w for w, i in w2_idx]
    
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
    targets = -sums1
    idx = np.searchsorted(sums2_sorted, targets)
    
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
    
    INF = np.iinfo(np.int64).max
    abs_c0[sum_c0 == 0] = INF
    abs_c1[sum_c1 == 0] = INF
    abs_c2[sum_c2 == 0] = INF
    
    min0 = np.min(abs_c0)
    min1 = np.min(abs_c1)
    min2 = np.min(abs_c2)
    
    best_abs = min(min0, min1, min2)
    
    if best_abs == INF:
        return [0]*30

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
        
    def find_coeffs(weights, target):
        n_w = len(weights)
        suffix_sums = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
        coeffs = [0] * n_w
        
        def dfs(idx, current_sum):
            if idx == n_w:
                return current_sum == target
            
            if abs(target - current_sum) > suffix_sums[idx]:
                return False
            
            w = weights[idx]
            
            # Try 0 first (most likely)
            coeffs[idx] = 0
            if dfs(idx + 1, current_sum):
                return True
            
            # Try 1
            coeffs[idx] = 1
            if dfs(idx + 1, current_sum + w):
                return True
            
            # Try -1
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
        return [0]*30
    
    # Map back to original positions
    result = [0] * n
    for coeff, (w, orig_idx) in zip(c1, w1_idx):
        result[orig_idx] = coeff
    for coeff, (w, orig_idx) in zip(c2, w2_idx):
        result[orig_idx] = coeff
    
    return result

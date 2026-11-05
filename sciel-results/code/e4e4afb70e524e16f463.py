import numpy as np
from math import gcd
from functools import reduce
import bisect

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    L = reduce(lcm, range(1, 31))
    weights = [L // i for i in range(1, 31)]
    
    wA = weights[:15]
    wB = weights[15:]
    
    def gen_sums(ws):
        sums = np.array([0], dtype=np.int64)
        for w in ws:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    sums_A = gen_sums(wA)
    sums_B = gen_sums(wB)
    sums_B_sorted = np.sort(sums_B)
    
    nB = len(sums_B_sorted)
    MAX_INT = np.iinfo(np.int64).max
    
    indices = np.searchsorted(sums_B_sorted, -sums_A)
    
    idx_right = indices
    idx_left = indices - 1
    
    mask_right = idx_right < nB
    mask_left = idx_left >= 0
    
    idx_right_clipped = np.clip(idx_right, 0, nB - 1)
    idx_left_clipped = np.clip(idx_left, 0, nB - 1)
    
    val_right = np.where(mask_right, sums_B_sorted[idx_right_clipped], MAX_INT)
    val_left = np.where(mask_left, sums_B_sorted[idx_left_clipped], MAX_INT)
    
    total_right = sums_A + val_right
    total_left = sums_A + val_left
    
    abs_right = np.abs(total_right)
    abs_left = np.abs(total_left)
    
    # Mask out zeros
    abs_right[total_right == 0] = MAX_INT
    abs_left[total_left == 0] = MAX_INT
    
    min_abs = np.minimum(abs_right, abs_left)
    best_idx = np.argmin(min_abs)
    
    if min_abs[best_idx] == MAX_INT:
        return [0] * 28 + [1, -1]
        
    if abs_right[best_idx] <= abs_left[best_idx]:
        best_sA = sums_A[best_idx]
        best_sB = val_right[best_idx]
    else:
        best_sA = sums_A[best_idx]
        best_sB = val_left[best_idx]
        
    def find_coeffs(ws, target):
        n = len(ws)
        if n == 0:
            return [] if target == 0 else None
        
        mid = n // 2
        w1 = ws[:mid]
        w2 = ws[mid:]
        
        sums1 = {0: []}
        for w in w1:
            new_sums = {}
            for s, c in sums1.items():
                new_sums[s - w] = c + [-1]
                new_sums[s] = c + [0]
                new_sums[s + w] = c + [1]
            sums1 = new_sums
            
        sums2 = {0: []}
        for w in w2:
            new_sums = {}
            for s, c in sums2.items():
                new_sums[s - w] = c + [-1]
                new_sums[s] = c + [0]
                new_sums[s + w] = c + [1]
            sums2 = new_sums
            
        keys2 = sorted(sums2.keys())
        vals2 = [sums2[k] for k in keys2]
        
        for s1, c1 in sums1.items():
            target2 = target - s1
            idx = bisect.bisect_left(keys2, target2)
            if idx < len(keys2) and keys2[idx] == target2:
                return c1 + vals2[idx]
            if idx > 0 and keys2[idx - 1] == target2:
                return c1 + vals2[idx - 1]
                
        return None

    coeffs_A = find_coeffs(wA, best_sA)
    coeffs_B = find_coeffs(wB, best_sB)
    
    if coeffs_A is None or coeffs_B is None:
        return [0] * 28 + [1, -1]
        
    return coeffs_A + coeffs_B

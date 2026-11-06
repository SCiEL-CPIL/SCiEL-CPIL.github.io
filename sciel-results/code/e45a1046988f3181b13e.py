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
    
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    mid = 14
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    sums1_sorted = np.sort(sums1)
    
    targets = -sums2
    
    len_s1 = len(sums1_sorted)
    
    # Find predecessor and successor indices
    idx_l = np.searchsorted(sums1_sorted, targets, side='left')
    idx_r = np.searchsorted(sums1_sorted, targets, side='right')
    
    # Candidate 1: Predecessor
    # idx_l - 1 is the largest index < target (if target exists) or largest < target (if not)
    # Wait, if target not in array, idx_l is first > target. So idx_l - 1 is last < target. Correct.
    # If target in array, idx_l is first == target. So idx_l - 1 is last < target. Correct.
    
    # We need to be careful with bounds.
    # Create candidate arrays with 0s for invalid, then mask them out.
    
    # Predecessor candidates
    idx_cand1 = idx_l - 1
    valid1 = idx_cand1 >= 0
    idx_cand1_clipped = np.clip(idx_cand1, 0, len_s1 - 1)
    s1_cand1 = sums1_sorted[idx_cand1_clipped]
    
    total1 = sums2 + s1_cand1
    abs_total1 = np.abs(total1)
    
    # Mask out invalid (idx < 0) and zero sums
    mask1 = valid1 & (total1 != 0)
    abs_total1[~mask1] = np.iinfo(np.int64).max
    
    # Successor candidates
    # idx_r is first > target.
    # If target in array, idx_r is first > target. Correct.
    # If target not in array, idx_r == idx_l, which is first > target. Correct.
    
    idx_cand2 = idx_r
    valid2 = idx_cand2 < len_s1
    idx_cand2_clipped = np.clip(idx_cand2, 0, len_s1 - 1)
    s1_cand2 = sums1_sorted[idx_cand2_clipped]
    
    total2 = sums2 + s1_cand2
    abs_total2 = np.abs(total2)
    
    mask2 = valid2 & (total2 != 0)
    abs_total2[~mask2] = np.iinfo(np.int64).max
    
    # Find global minimum
    min_abs1 = np.min(abs_total1)
    min_abs2 = np.min(abs_total2)
    
    if min_abs1 < min_abs2:
        best_abs = min_abs1
        idx_best_in_1 = np.argmin(abs_total1)
        s2_val = sums2[idx_best_in_1]
        s1_val = s1_cand1[idx_best_in_1]
    else:
        best_abs = min_abs2
        idx_best_in_2 = np.argmin(abs_total2)
        s2_val = sums2[idx_best_in_2]
        s1_val = s1_cand2[idx_best_in_2]
        
    # Reconstruct coefficients
    def reconstruct_coeffs(weights, target_sum):
        # DFS with pruning
        n_w = len(weights)
        # Sort weights and track indices? No, just return coeffs in original order.
        # It's easier to solve for the specific weights list.
        # Weights are distinct enough that ordering matters for pruning speed.
        # Let's sort weights by magnitude descending for the search, then map back.
        order = np.argsort(-weights)
        sorted_weights = weights[order]
        
        # Suffix sums
        suffix_max = np.zeros(n_w + 1, dtype=np.int64)
        suffix_min = np.zeros(n_w + 1, dtype=np.int64)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + sorted_weights[i]
            suffix_min[i] = suffix_min[i+1] - sorted_weights[i]
            
        coeffs = [0] * n_w
        
        def dfs(i, current_sum):
            if i == n_w:
                if current_sum == target_sum:
                    return True
                return False
            
            # Pruning
            # Max possible add: suffix_max[i]
            # Min possible add: suffix_min[i]
            if current_sum + suffix_max[i] < target_sum:
                return False
            if current_sum + suffix_min[i] > target_sum:
                return False
                
            # Try -1
            coeffs[i] = -1
            if dfs(i + 1, current_sum - sorted_weights[i]):
                return True
            # Try 0
            coeffs[i] = 0
            if dfs(i + 1, current_sum):
                return True
            # Try 1
            coeffs[i] = 1
            if dfs(i + 1, current_sum + sorted_weights[i]):
                return True
                
            return False
        
        if dfs(0, 0):
            # Map back to original order
            orig_coeffs = [0] * n_w
            for i in range(n_w):
                orig_idx = order[i]
                orig_coeffs[orig_idx] = coeffs[i]
            return orig_coeffs
        else:
            # Should not happen
            return [0] * n_w

    c1 = reconstruct_coeffs(w1, s1_val)
    c2 = reconstruct_coeffs(w2, s2_val)
    
    final_coeffs = c1 + c2
    
    # Verification
    # val = sum(c[i] / (i+1) for i in range(n))
    # assert val != 0
    
    return final_coeffs

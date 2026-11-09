import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # LCM of 2..48
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Precompute numerator for each denominator
    vals = [lcm_val // d for d in denominators]
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    indices1 = list(range(n1))
    indices2 = list(range(n1, n))
    
    vals1 = [vals[i] for i in indices1]
    vals2 = [vals[i] for i in indices2]
    
    # Generate sums for half 1
    sums1_sums = [0]
    sums1_masks = [0]
    for i in range(n1):
        v = vals1[i]
        bit = 1 << i
        current_len = len(sums1_sums)
        for j in range(current_len):
            sums1_sums.append(sums1_sums[j] + v)
            sums1_masks.append(sums1_masks[j] | bit)
            
    # Sort sums1 by sum
    pairs = list(zip(sums1_sums, sums1_masks))
    pairs.sort(key=lambda x: x[0])
    sums1_sums = [p[0] for p in pairs]
    sums1_masks = [p[1] for p in pairs]
    
    max_s1 = sums1_sums[-1]
    
    # Prepare half 2 for DFS
    # Sort by value descending to find good solutions early
    items2 = [(vals2[i], i) for i in range(n2)]
    items2.sort(key=lambda x: x[0], reverse=True)
    sorted_vals2 = [v for v, i in items2]
    sorted_indices2 = [i for v, i in items2]
    
    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0
    
    # DFS for half 2
    # idx: current index in sorted_vals2
    # curr_sum: sum of selected values in half 2 so far
    # curr_mask: bitmask of selected values in half 2 so far
    def dfs(idx, curr_sum, curr_mask):
        nonlocal best_sum, best_mask1, best_mask2
        
        if time.time() > deadline:
            return
            
        # Pruning: even if we take all remaining, can we beat best?
        # We don't have a precomputed suffix sum for half 2 easily in this structure.
        # But we have max_s1.
        # Upper bound for this branch: curr_sum + sum(remaining_vals2) + max_s1
        # We can precompute suffix sums for sorted_vals2.
        
        if idx == n2:
            target = lcm_val - 1 - curr_sum
            if target >= 0:
                b_idx = bisect.bisect_right(sums1_sums, target)
                if b_idx > 0:
                    s1 = sums1_sums[b_idx - 1]
                    m1 = sums1_masks[b_idx - 1]
                    total = s1 + curr_sum
                    if total > best_sum:
                        best_sum = total
                        best_mask1 = m1
                        best_mask2 = curr_mask
            return
        
        # Prune if curr_sum + max_possible_remaining + max_s1 <= best_sum
        # max_possible_remaining is suffix_sum[idx]
        
        # Exclude
        dfs(idx + 1, curr_sum, curr_mask)
        
        # Include
        dfs(idx + 1, curr_sum + sorted_vals2[idx], curr_mask | (1 << idx))
        
    # Precompute suffix sums for sorted_vals2 for pruning
    suffix_sums = [0] * (n2 + 1)
    for i in range(n2 - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_vals2[i]
        
    # Modify dfs to use suffix_sums
    def dfs_pruned(idx, curr_sum, curr_mask):
        nonlocal best_sum, best_mask1, best_mask2
        
        if time.time() > deadline:
            return
            
        # Pruning
        if curr_sum + suffix_sums[idx] + max_s1 <= best_sum:
            return
            
        if idx == n2:
            target = lcm_val - 1 - curr_sum
            if target >= 0:
                b_idx = bisect.bisect_right(sums1_sums, target)
                if b_idx > 0:
                    s1 = sums1_sums[b_idx - 1]
                    m1 = sums1_masks[b_idx - 1]
                    total = s1 + curr_sum
                    if total > best_sum:
                        best_sum = total
                        best_mask1 = m1
                        best_mask2 = curr_mask
            return
        
        # Exclude
        dfs_pruned(idx + 1, curr_sum, curr_mask)
        
        # Include
        dfs_pruned(idx + 1, curr_sum + sorted_vals2[idx], curr_mask | (1 << idx))
        
    dfs_pruned(0, 0, 0)
    
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            result_indices.append(indices1[i])
    for i in range(n2):
        if best_mask2 & (1 << i):
            # Map back to original index in indices2
            # sorted_indices2[i] is the index in the original half2 list
            orig_idx_in_half2 = sorted_indices2[i]
            result_indices.append(indices2[orig_idx_in_half2])
            
    return result_indices

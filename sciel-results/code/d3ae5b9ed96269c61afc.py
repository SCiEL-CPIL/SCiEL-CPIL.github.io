import time
from math import lcm
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights
    weights = [L // d for d in denominators]
    
    # Meet-in-the-middle
    mid = 23
    n1 = mid
    n2 = n - mid
    
    # Generate sums for first half
    # We store (sum, mask)
    sums1 = [(0, 0)]
    for i in range(n1):
        w = weights[i]
        bit = 1 << i
        # We iterate over a copy of the current state to avoid modifying while iterating
        # But standard doubling is:
        # current_len = len(sums1)
        # for k in range(current_len):
        #   s, m = sums1[k]
        #   ns = s + w
        #   if ns < L: sums1.append((ns, m | bit))
        
        # To be faster, we can pre-allocate or just use extend
        new_items = []
        for s, m in sums1:
            ns = s + w
            if ns < L:
                new_items.append((ns, m | bit))
        sums1.extend(new_items)
        
        if time.time() > deadline - 30:
            break

    # Generate sums for second half
    sums2 = [(0, 0)]
    for i in range(n2):
        w = weights[mid + i]
        bit = 1 << i
        new_items = []
        for s, m in sums2:
            ns = s + w
            if ns < L:
                new_items.append((ns, m | bit))
        sums2.extend(new_items)
        
        if time.time() > deadline - 30:
            break

    # Sort second half by sum
    # We only need the sums for bisect, but we need masks to reconstruct
    # So we sort a list of indices or tuples
    # Tuples (sum, mask) sort by sum then mask. This is fine.
    sums2.sort()
    
    # Extract sums for bisect
    sums2_vals = [s for s, m in sums2]
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    # For each sum in first half, find best in second half
    for s1, m1 in sums1:
        if time.time() > deadline - 10:
            break
            
        target = L - s1 - 1
        # Find rightmost index in sums2_vals <= target
        pos = bisect.bisect_right(sums2_vals, target)
        if pos > 0:
            s2, m2 = sums2[pos - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2

    # Reconstruct best subset
    result_indices = []
    for i in range(n1):
        if best_m1 & (1 << i):
            result_indices.append(i)
    for i in range(n2):
        if best_m2 & (1 << i):
            result_indices.append(mid + i)
    
    if not result_indices:
        return [0]
    
    # Local Search with Integer Arithmetic
    # We want to maximize Sum < L
    # Current sum is best_sum
    current_subset = set(result_indices)
    current_sum = best_sum
    
    # Pre-compute weights for fast access
    # Weights are already in 'weights'
    
    def try_improve(subset, sum_val, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            in_sub = list(subset)
            not_in_sub = [i for i in range(n) if i not in subset]
            
            # 1. Try adding one element
            best_add_gain = 0
            best_add_idx = -1
            for j in not_in_sub:
                w_j = weights[j]
                if sum_val + w_j < L:
                    gain = w_j
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_idx = j
            
            if best_add_idx != -1:
                subset.add(best_add_idx)
                sum_val += best_add_gain
                improved = True
                continue # Restart loop as set changed significantly
            
            # 2. Try swapping one element
            best_swap_gain = 0
            best_swap_out = -1
            best_swap_in = -1
            for i in in_sub:
                w_i = weights[i]
                for j in not_in_sub:
                    w_j = weights[j]
                    new_sum = sum_val - w_i + w_j
                    if new_sum < L:
                        gain = w_j - w_i
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_out = i
                            best_swap_in = j
            
            if best_swap_in != -1:
                subset.remove(best_swap_out)
                subset.add(best_swap_in)
                sum_val += best_swap_gain
                improved = True
                continue
            
            # 3. Try 2-out 1-in
            # This is O(N^3) which might be slow if N is large, but N=47.
            # 47^3 ~ 100k. Feasible.
            best_21_gain = 0
            best_21_out1 = -1
            best_21_out2 = -1
            best_21_in = -1
            
            # Optimization: Only consider swaps if potential gain is positive
            # w_j > w_i1 + w_i2
            for i1_idx in range(len(in_sub)):
                i1 = in_sub[i1_idx]
                w_i1 = weights[i1]
                for i2_idx in range(i1_idx + 1, len(in_sub)):
                    i2 = in_sub[i2_idx]
                    w_i2 = weights[i2]
                    for j in not_in_sub:
                        w_j = weights[j]
                        new_sum = sum_val - w_i1 - w_i2 + w_j
                        if new_sum < L:
                            gain = w_j - w_i1 - w_i2
                            if gain > best_21_gain:
                                best_21_gain = gain
                                best_21_out1 = i1
                                best_21_out2 = i2
                                best_21_in = j
            
            if best_21_in != -1:
                subset.remove(best_21_out1)
                subset.remove(best_21_out2)
                subset.add(best_21_in)
                sum_val += best_21_gain
                improved = True
                continue
            
            # 4. Try 1-out 2-in
            best_12_gain = 0
            best_12_out = -1
            best_12_in1 = -1
            best_12_in2 = -1
            
            for i in in_sub:
                w_i = weights[i]
                for j1_idx in range(len(not_in_sub)):
                    j1 = not_in_sub[j1_idx]
                    w_j1 = weights[j1]
                    for j2_idx in range(j1_idx + 1, len(not_in_sub)):
                        j2 = not_in_sub[j2_idx]
                        w_j2 = weights[j2]
                        new_sum = sum_val - w_i + w_j1 + w_j2
                        if new_sum < L:
                            gain = w_j1 + w_j2 - w_i
                            if gain > best_12_gain:
                                best_12_gain = gain
                                best_12_out = i
                                best_12_in1 = j1
                                best_12_in2 = j2
            
            if best_12_in1 != -1:
                subset.remove(best_12_out)
                subset.add(best_12_in1)
                subset.add(best_12_in2)
                sum_val += best_12_gain
                improved = True
                continue

        return subset, sum_val

    if time.time() < deadline - 5:
        current_subset, current_sum = try_improve(current_subset, current_sum, deadline - 2)
    
    # Final check
    result_indices = list(current_subset)
    if not result_indices:
        result_indices = [0]
        
    return result_indices

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import numpy as np
import bisect

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin: kill at 90s, use 55s for computation
    
    n = 30
    L = _lcm_range(n)
    weights = [L // (i + 1) for i in range(n)]
    
    # Trivial fallback: 1/29 - 1/30 = 1/870
    best_coeffs = [0] * 28 + [1, -1]
    best_N = weights[28] - weights[29]
    best_abs_N = abs(best_N)
    
    # Split into two halves of 15
    split_idx = 15
    core_weights = weights[:split_idx]
    tail_weights = weights[split_idx:]
    
    # Generate all possible sums for each half using iterative expansion
    # 3^15 = 14,348,907 states each - this is the bottleneck
    # Use numpy to speed up where possible, but sets are hard to vectorize
    # Alternative: generate sums as a list and sort
    
    def generate_sums(weights_list):
        """Generate all possible sums for given weights."""
        sums = [0]
        for w in weights_list:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
            if time.time() > deadline - 10:
                break
        return sums
    
    # Check time before starting expensive computation
    if time.time() > deadline - 40:
        return best_coeffs
    
    # Generate core sums
    core_sums_list = generate_sums(core_weights)
    if time.time() > deadline - 15:
        return best_coeffs
    
    # Generate tail sums
    tail_sums_list = generate_sums(tail_weights)
    if time.time() > deadline - 10:
        return best_coeffs
    
    # Convert to sets for O(1) lookup and also keep sorted lists
    core_sums_set = set(core_sums_list)
    tail_sums_set = set(tail_sums_list)
    
    # Sort for binary search
    sorted_core = sorted(core_sums_list)
    sorted_tail = sorted(tail_sums_list)
    
    # Find the best (C, T) pair minimizing |C + T| != 0
    # Strategy: iterate over the smaller set and binary search in the other
    # Both are ~14M elements
    
    best_K_val = best_abs_N
    best_C_found = None
    best_T_found = None
    
    # Iterate over tail sums and find closest core sum to -T
    # To speed up, we can limit the search if we find a very small value
    
    len_core = len(sorted_core)
    len_tail = len(sorted_tail)
    
    # Use the smaller set for iteration
    if len_tail <= len_core:
        iter_sums = tail_sums_list
        search_sorted = sorted_core
        search_set = core_sums_set
        is_tail_iter = True
    else:
        iter_sums = core_sums_list
        search_sorted = sorted_tail
        search_set = tail_sums_set
        is_tail_iter = False
    
    found_optimal = False
    
    for i, T in enumerate(iter_sums):
        if is_tail_iter:
            target = -T
        else:
            target = -T
        
        # Binary search for target in search_sorted
        pos = bisect.bisect_left(search_sorted, target)
        
        candidates = []
        if pos < len(search_sorted):
            candidates.append(search_sorted[pos])
        if pos > 0:
            candidates.append(search_sorted[pos - 1])
        
        for C in candidates:
            total = C + T
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_K_val:
                best_K_val = abs_total
                if is_tail_iter:
                    best_T_found = T
                    best_C_found = C
                else:
                    best_C_found = T  # T is from iter_sums which is core
                    best_T_found = C  # C is from search_sorted which is tail
                if best_K_val == 1:
                    found_optimal = True
                    break
        
        if found_optimal:
            break
        
        # Time check every 100000 iterations
        if i % 100000 == 0 and time.time() > deadline - 10:
            break
    
    # If we found a better solution, reconstruct coefficients
    if best_K_val < best_abs_N and best_C_found is not None and best_T_found is not None:
        # Reconstruct coefficients using DFS with pruning
        def find_coeffs(weights, target):
            coeffs = [0] * len(weights)
            
            # Precompute suffix sums for pruning
            suffix_sums = [0] * (len(weights) + 1)
            for i in range(len(weights) - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + abs(weights[i])
            
            def dfs(idx, current_sum):
                if idx == len(weights):
                    return current_sum == target
                # Pruning
                if current_sum + suffix_sums[idx] < target:
                    return False
                if current_sum - suffix_sums[idx] > target:
                    return False
                
                for val in (-1, 0, 1):
                    coeffs[idx] = val
                    if dfs(idx + 1, current_sum + val * weights[idx]):
                        return True
                return False
            
            if dfs(0, 0):
                return coeffs
            return None
        
        # Determine which is core and which is tail
        if is_tail_iter:
            core_target = best_C_found
            tail_target = best_T_found
        else:
            core_target = best_T_found  # search_sorted was tail
            tail_target = best_C_found  # iter_sums was core... wait
        
        # Let me re-think the mapping
        # If is_tail_iter: iter_sums = tail, search = core
        #   So T is from tail, C is from core
        #   best_T_found = T (tail sum), best_C_found = C (core sum)
        # If not is_tail_iter: iter_sums = core, search = tail
        #   So T is from core, C is from tail
        #   best_T_found = C (tail sum), best_C_found = T (core sum)
        
        if is_tail_iter:
            core_target = best_C_found
            tail_target = best_T_found
        else:
            core_target = best_T_found  # This is actually the core sum
            tail_target = best_C_found  # This is actually the tail sum
        
        # Wait, I need to be more careful
        # Let me just always assign:
        # If is_tail_iter: iter_sums=tail, search=core
        #   For each T in tail, find C in core close to -T
        #   best_T_found = T (tail), best_C_found = C (core)
        # If not is_tail_iter: iter_sums=core, search=tail  
        #   For each T in core, find C in tail close to -T
        #   best_T_found = T (core), best_C_found = C (tail)
        
        if is_tail_iter:
            core_target = best_C_found
            tail_target = best_T_found
        else:
            core_target = best_T_found
            tail_target = best_C_found
        
        core_coeffs = find_coeffs(core_weights, core_target)
        tail_coeffs = find_coeffs(tail_weights, tail_target)
        
        if core_coeffs is not None and tail_coeffs is not None:
            new_coeffs = core_coeffs + tail_coeffs
            check_N = sum(c * w for c, w in zip(new_coeffs, weights))
            if check_N != 0 and abs(check_N) <= best_K_val:
                best_coeffs = new_coeffs
                best_N = check_N
                best_abs_N = abs(check_N)
    
    # Local search: try flipping individual coefficients to improve
    if best_abs_N > 1:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                old_c = best_coeffs[i]
                for new_c in (-1, 0, 1):
                    if new_c == old_c:
                        continue
                    best_coeffs[i] = new_c
                    new_N = sum(c * w for c, w in zip(best_coeffs, weights))
                    if new_N == 0:
                        best_coeffs[i] = old_c
                        continue
                    if abs(new_N) < best_abs_N:
                        best_abs_N = abs(new_N)
                        best_N = new_N
                        improved = True
                    else:
                        best_coeffs[i] = old_c
    
    return best_coeffs

# EVOLVE-BLOCK-END

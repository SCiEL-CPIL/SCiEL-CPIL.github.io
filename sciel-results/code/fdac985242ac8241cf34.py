# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    reciprocals = [1.0 / numbers[i] for i in range(n)]
    
    # Precompute conflict bitmasks
    conflict_mask = [0] * n
    for i in range(n):
        ai = numbers[i]
        for j in range(i + 1, n):
            aj = numbers[j]
            g = gcd(ai, aj)
            if ai * aj // g <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def can_add(idx, selected_mask):
        return (conflict_mask[idx] & selected_mask) == 0
    
    def reciprocal_sum_from_mask(selected_mask):
        s = 0.0
        for i in range(n):
            if (selected_mask >> i) & 1:
                s += reciprocals[i]
        return s
    
    def greedy_construct(order):
        selected_mask = 0
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask |= (1 << idx)
        return selected_mask
    
    def local_search(selected_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_sum = reciprocal_sum_from_mask(selected_mask)
            
            # Try adding any single element
            for idx in range(n):
                if not ((selected_mask >> idx) & 1):
                    if can_add(idx, selected_mask):
                        selected_mask |= (1 << idx)
                        current_sum += reciprocals[idx]
                        improved = True
                        break
            
            if time.time() >= time_limit:
                break
            
            # Try 1-for-1 swap: remove i (in set), add j (not in set)
            if not improved:
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if not ((selected_mask >> i) & 1):
                        continue
                    # i is in the set; try removing it
                    mask_without_i = selected_mask & ~(1 << i)
                    best_j = -1
                    best_gain = 0.0
                    for j in range(n):
                        if ((selected_mask >> j) & 1):
                            continue
                        # j is not in set; check if j can be added after removing i
                        if can_add(j, mask_without_i):
                            gain = reciprocals[j] - reciprocals[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                    if best_j >= 0 and best_gain > 1e-15:
                        selected_mask = mask_without_i | (1 << best_j)
                        current_sum += best_gain
                        improved = True
            
            # Try 2-for-2 swap: remove i1,i2 (in set), add j1,j2 (not in set)
            if not improved and time.time() < time_limit:
                best_swap = None
                best_gain = 0.0
                in_set = [i for i in range(n) if (selected_mask >> i) & 1]
                not_in_set = [j for j in range(n) if not ((selected_mask >> j) & 1)]
                
                for i1_idx in range(len(in_set)):
                    if time.time() >= time_limit:
                        break
                    i1 = in_set[i1_idx]
                    for i2_idx in range(i1_idx + 1, len(in_set)):
                        i2 = in_set[i2_idx]
                        mask_without_pair = selected_mask & ~(1 << i1) & ~(1 << i2)
                        base_gain = -reciprocals[i1] - reciprocals[i2]
                        
                        for j1 in not_in_set:
                            if not can_add(j1, mask_without_pair):
                                continue
                            for j2 in not_in_set:
                                if j2 <= j1:
                                    continue
                                if not can_add(j2, mask_without_pair | (1 << j1)):
                                    continue
                                gain = base_gain + reciprocals[j1] + reciprocals[j2]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_swap = (i1, i2, j1, j2)
                
                if best_swap is not None and best_gain > 1e-15:
                    i1, i2, j1, j2 = best_swap
                    selected_mask = (selected_mask & ~(1 << i1) & ~(1 << i2)) | (1 << j1) | (1 << j2)
                    current_sum += best_gain
                    improved = True
            
            # Try adding two elements simultaneously
            if not improved and time.time() < time_limit:
                best_pair = (-1, -1)
                best_gain = 0.0
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if ((selected_mask >> i) & 1):
                        continue
                    if not can_add(i, selected_mask):
                        continue
                    for j in range(i + 1, n):
                        if ((selected_mask >> j) & 1):
                            continue
                        if not can_add(j, selected_mask | (1 << i)):
                            continue
                        gain = reciprocals[i] + reciprocals[j]
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (i, j)
                if best_pair != (-1, -1) and best_gain > 1e-15:
                    i, j = best_pair
                    selected_mask |= (1 << i) | (1 << j)
                    current_sum += best_gain
                    improved = True
        
        return selected_mask
    
    def verify(result):
        """Verify all pairs have lcm > 256."""
        for idx_a in range(len(result)):
            for idx_b in range(idx_a + 1, len(result)):
                a, b = result[idx_a], result[idx_b]
                if a * b // gcd(a, b) <= N:
                    return False
        return True
    
    def local_search_with_core(core_mask, time_limit):
        """
        Perform local search starting from a solution that must include the core.
        The core elements are fixed; we only optimize the rest.
        """
        # Start with the core
        selected_mask = core_mask
        
        # Greedily add non-conflicting elements not in core
        for idx in range(n):
            if not ((core_mask >> idx) & 1) and can_add(idx, selected_mask):
                selected_mask |= (1 << idx)
        
        # Now do local search, but never remove core elements
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_sum = reciprocal_sum_from_mask(selected_mask)
            
            # Try adding any single non-core element
            for idx in range(n):
                if not ((selected_mask >> idx) & 1) and not ((core_mask >> idx) & 1):
                    if can_add(idx, selected_mask):
                        selected_mask |= (1 << idx)
                        current_sum += reciprocals[idx]
                        improved = True
                        break
            
            if time.time() >= time_limit:
                break
            
            # Try 1-for-1 swap: remove i (in set, not core), add j (not in set, not core)
            if not improved:
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if not ((selected_mask >> i) & 1):
                        continue
                    if (core_mask >> i) & 1:
                        continue  # Can't remove core elements
                    mask_without_i = selected_mask & ~(1 << i)
                    best_j = -1
                    best_gain = 0.0
                    for j in range(n):
                        if ((selected_mask >> j) & 1):
                            continue
                        if (core_mask >> j) & 1:
                            continue  # Core elements are already in
                        if can_add(j, mask_without_i):
                            gain = reciprocals[j] - reciprocals[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                    if best_j >= 0 and best_gain > 1e-15:
                        selected_mask = mask_without_i | (1 << best_j)
                        current_sum += best_gain
                        improved = True
            
            # Try 2-for-2 swap
            if not improved and time.time() < time_limit:
                best_swap = None
                best_gain = 0.0
                in_set_non_core = [i for i in range(n) if (selected_mask >> i) & 1 and not ((core_mask >> i) & 1)]
                not_in_set_non_core = [j for j in range(n) if not ((selected_mask >> j) & 1) and not ((core_mask >> j) & 1)]
                
                for i1_idx in range(len(in_set_non_core)):
                    if time.time() >= time_limit:
                        break
                    i1 = in_set_non_core[i1_idx]
                    for i2_idx in range(i1_idx + 1, len(in_set_non_core)):
                        i2 = in_set_non_core[i2_idx]
                        mask_without_pair = selected_mask & ~(1 << i1) & ~(1 << i2)
                        base_gain = -reciprocals[i1] - reciprocals[i2]
                        
                        for j1 in not_in_set_non_core:
                            if not can_add(j1, mask_without_pair):
                                continue
                            for j2 in not_in_set_non_core:
                                if j2 <= j1:
                                    continue
                                if not can_add(j2, mask_without_pair | (1 << j1)):
                                    continue
                                gain = base_gain + reciprocals[j1] + reciprocals[j2]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_swap = (i1, i2, j1, j2)
                
                if best_swap is not None and best_gain > 1e-15:
                    i1, i2, j1, j2 = best_swap
                    selected_mask = (selected_mask & ~(1 << i1) & ~(1 << i2)) | (1 << j1) | (1 << j2)
                    current_sum += best_gain
                    improved = True
        
        return selected_mask
    
    best_mask = 0
    best_sum = 0.0
    
    # Strategy 1: No core (original approach)
    order_sorted = list(range(n))
    mask = greedy_construct(order_sorted)
    mask = local_search(mask, deadline - 3.0)
    s = reciprocal_sum_from_mask(mask)
    if s > best_sum:
        best_sum = s
        best_mask = mask
    
    # Strategy 2: Forced cores with single small numbers
    # Small numbers have high reciprocal value, so forcing them may help
    # but we need to check if they conflict with each other
    small_numbers = [i for i in range(n) if numbers[i] <= 20]
    
    for core_idx in small_numbers:
        if time.time() >= deadline - 2.0:
            break
        
        core_mask = 1 << core_idx
        
        # Check if this core is valid (it's a single element, so always valid)
        mask = local_search_with_core(core_mask, min(time.time() + 0.5, deadline - 1.5))
        s = reciprocal_sum_from_mask(mask)
        if s > best_sum:
            best_sum = s
            best_mask = mask
    
    # Strategy 3: Forced cores with pairs of small numbers
    # Only consider pairs that don't conflict
    for i_idx in range(len(small_numbers)):
        if time.time() >= deadline - 1.0:
            break
        i = small_numbers[i_idx]
        for j_idx in range(i_idx + 1, len(small_numbers)):
            j = small_numbers[j_idx]
            # Check if i and j conflict
            if (conflict_mask[i] >> j) & 1:
                continue
            
            core_mask = (1 << i) | (1 << j)
            
            mask = local_search_with_core(core_mask, min(time.time() + 0.3, deadline - 1.0))
            s = reciprocal_sum_from_mask(mask)
            if s > best_sum:
                best_sum = s
                best_mask = mask
    
    # Randomized restarts with diverse orderings (remaining time)
    trial = 0
    while time.time() < deadline - 1.0:
        trial += 1
        
        strategy = trial % 4
        if strategy == 0:
            order = list(range(n))
            random.shuffle(order)
        elif strategy == 1:
            order = sorted(range(n), key=lambda i: reciprocals[i] / (1.0 + bin(conflict_mask[i]).count('1')), reverse=True)
        elif strategy == 2:
            order = list(range(n))
            random.shuffle(order)
        else:
            order = sorted(range(n), key=lambda i: (-reciprocals[i], bin(conflict_mask[i]).count('1')))
        
        mask = greedy_construct(order)
        remaining = deadline - 0.5 - time.time()
        if remaining > 0.1:
            mask = local_search(mask, time.time() + max(0.1, remaining * 0.7))
        s = reciprocal_sum_from_mask(mask)
        if s > best_sum:
            best_sum = s
            best_mask = mask
        
        if trial >= 40:
            break
    
    # Final polish
    remaining = deadline - 0.3 - time.time()
    if remaining > 0.1:
        best_mask = local_search(best_mask, time.time() + remaining)
    
    # Extract result
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(numbers[i])
    
    # Verify and fix if needed
    if result and not verify(result):
        result = [251]
    
    if not result:
        result = [251]
    
    return result
# EVOLVE-BLOCK-END

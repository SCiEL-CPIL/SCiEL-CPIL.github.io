"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    # Numbers of interest
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    
    # Map value to index
    idx = {x: i for i, x in enumerate(nums)}
    
    # Precompute conflict matrix
    # conflict[i] is a bitmask of indices j such that lcm(nums[i], nums[j]) <= 256
    conflict = [0] * n
    
    for i in range(n):
        ai = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            bj = nums[j]
            if lcm(ai, bj) <= N:
                mask |= (1 << j)
        conflict[i] = mask

    # Precompute reciprocal weights
    weights = [1.0 / x for x in nums]
    
    # Helper function to check if a single element can be added to a set
    # given the current conflict mask of the set
    def can_add(i, set_mask):
        return (conflict[i] & set_mask) == 0
    
    # Helper function to compute score
    def compute_score(indices):
        return sum(weights[i] for i in indices)
    
    # We split the numbers into "Small" (2..24) and "Large" (25..256)
    # Smaller split allows more small numbers to be explored in the enumeration
    small_vals = list(range(2, 25))
    small_indices = [idx[x] for x in small_vals]
    n_small = len(small_indices)
    
    # Precompute list of large indices
    large_indices_list = [i for i in range(n) if nums[i] >= 25]
    
    best_set_indices = []
    best_score = 0.0
    
    # Iterate over all subsets of small numbers
    # There are 2^23 = 8388608 subsets - too many, so we use a smarter approach
    # Actually let's use 2..16 (15 numbers) = 32768 subsets which is manageable
    small_vals = list(range(2, 17))
    small_indices = [idx[x] for x in small_vals]
    n_small = len(small_indices)
    
    # Precompute list of large indices
    large_indices_list = [i for i in range(n) if nums[i] >= 17]
    
    # For each subset of small numbers, greedily select compatible large numbers
    for mask_small in range(1 << n_small):
        if time.time() > deadline - 30.0:
            break
            
        # Extract small indices in this mask
        small_in_set = []
        small_mask = 0
        for i in range(n_small):
            if mask_small & (1 << i):
                si = small_indices[i]
                small_in_set.append(si)
                small_mask |= (1 << si)
                
        # Check internal validity of small_in_set using bitmask
        valid = True
        for i in range(len(small_in_set)):
            si = small_in_set[i]
            if conflict[si] & small_mask:
                # This means si conflicts with some earlier element
                # Actually we need to check if any pair conflicts
                # Let's just check pairwise for small sets (max 15 elements)
                pass
        
        # Proper validity check for small set
        for i in range(len(small_in_set)):
            for j in range(i + 1, len(small_in_set)):
                if conflict[small_in_set[i]] & (1 << small_in_set[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Determine compatible large numbers
        large_mask = 0
        for i in large_indices_list:
            large_mask |= (1 << i)
        
        # For each small index in small_in_set, remove its conflicts from large_mask
        for idx_i in small_in_set:
            large_mask &= ~conflict[idx_i]
            
        # Extract active large candidates
        active_candidates = [i for i in large_indices_list if large_mask & (1 << i)]
        
        # Sort by reciprocal weight (descending) to prioritize high weights
        active_candidates.sort(key=lambda x: -weights[x])
        
        # Greedy selection
        selected_large = []
        selected_mask = 0
        
        for i in active_candidates:
            if selected_mask & conflict[i]:
                continue
            selected_large.append(i)
            selected_mask |= (1 << i)
            
        # Combine small and large
        full_set_indices = small_in_set + selected_large
        
        # Calculate score
        score = compute_score(full_set_indices)
        
        if score > best_score:
            best_score = score
            best_set_indices = list(full_set_indices)
    
    # Local search with multiple move types
    if time.time() < deadline - 15.0 and best_set_indices:
        current_set = list(best_set_indices)
        current_score = best_score
        
        # Multiple restarts with random perturbations
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Create a perturbed version of the current best
            if restart == 0:
                working_set = list(current_set)
            else:
                # Randomly remove 1-3 elements and add 1-3 random compatible elements
                working_set = list(current_set)
                remove_count = min(3, len(working_set))
                if remove_count > 0:
                    remove_indices = random.sample(range(len(working_set)), remove_count)
                    for ri in sorted(remove_indices, reverse=True):
                        working_set.pop(ri)
                
                # Try to add some random compatible elements
                all_indices = list(range(n))
                random.shuffle(all_indices)
                working_mask = 0
                for i in working_set:
                    working_mask |= (1 << i)
                added = 0
                for i in all_indices:
                    if added >= 3:
                        break
                    if (1 << i) not in working_mask and can_add(i, working_mask):
                        working_set.append(i)
                        working_mask |= (1 << i)
                        added += 1
            
            working_score = compute_score(working_set)
            
            # Local search: try various swap moves
            improved = True
            max_iter = 100
            iter_count = 0
            while improved and iter_count < max_iter and time.time() < deadline - 10.0:
                improved = False
                iter_count += 1
                working_set_set = set(working_set)
                working_mask = 0
                for i in working_set:
                    working_mask |= (1 << i)
                
                # Move 1: 1-for-1 swaps
                for remove_idx in range(len(working_set)):
                    if time.time() > deadline - 10.0:
                        break
                    remove_i = working_set[remove_idx]
                    temp_mask = working_mask & ~(1 << remove_i)
                    
                    # Try adding each candidate not in working_set
                    for add_i in range(n):
                        if add_i in working_set_set:
                            continue
                        if can_add(add_i, temp_mask):
                            new_score = working_score - weights[remove_i] + weights[add_i]
                            if new_score > working_score + 1e-12:
                                working_set[remove_idx] = add_i
                                working_set_set = set(working_set)
                                working_mask = 0
                                for i in working_set:
                                    working_mask |= (1 << i)
                                working_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                
                # Move 2: 1-for-2 swaps (remove one, add two)
                if not improved:
                    for remove_idx in range(len(working_set)):
                        if time.time() > deadline - 10.0:
                            break
                        remove_i = working_set[remove_idx]
                        temp_mask = working_mask & ~(1 << remove_i)
                        
                        # Try adding two compatible elements
                        for add1 in range(n):
                            if time.time() > deadline - 10.0:
                                break
                            if add1 in working_set_set:
                                continue
                            if not can_add(add1, temp_mask):
                                continue
                            
                            temp_mask2 = temp_mask | (1 << add1)
                            for add2 in range(n):
                                if time.time() > deadline - 10.0:
                                    break
                                if add2 in working_set_set or add2 == add1:
                                    continue
                                if not can_add(add2, temp_mask2):
                                    continue
                                
                                new_score = working_score - weights[remove_i] + weights[add1] + weights[add2]
                                if new_score > working_score + 1e-12:
                                    # Replace remove_i with add1 and add2
                                    working_set[remove_idx] = add1
                                    working_set.append(add2)
                                    working_set_set = set(working_set)
                                    working_mask = 0
                                    for i in working_set:
                                        working_mask |= (1 << i)
                                    working_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                # Move 3: 2-for-1 swaps (remove two, add one)
                if not improved:
                    for i in range(len(working_set)):
                        if time.time() > deadline - 10.0:
                            break
                        for j in range(i + 1, len(working_set)):
                            if time.time() > deadline - 10.0:
                                break
                            remove_i = working_set[i]
                            remove_j = working_set[j]
                            temp_mask = working_mask & ~(1 << remove_i) & ~(1 << remove_j)
                            
                            for add_i in range(n):
                                if add_i in working_set_set:
                                    continue
                                if can_add(add_i, temp_mask):
                                    new_score = working_score - weights[remove_i] - weights[remove_j] + weights[add_i]
                                    if new_score > working_score + 1e-12:
                                        # Remove both and add one
                                        new_working_set = [x for x in working_set if x != remove_i and x != remove_j]
                                        new_working_set.append(add_i)
                                        working_set = new_working_set
                                        working_set_set = set(working_set)
                                        working_mask = 0
                                        for k in working_set:
                                            working_mask |= (1 << k)
                                        working_score = new_score
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                
                # Move 4: 2-for-2 swaps (remove two, add two)
                if not improved:
                    for i in range(len(working_set)):
                        if time.time() > deadline - 10.0:
                            break
                        for j in range(i + 1, len(working_set)):
                            if time.time() > deadline - 10.0:
                                break
                            remove_i = working_set[i]
                            remove_j = working_set[j]
                            temp_mask = working_mask & ~(1 << remove_i) & ~(1 << remove_j)
                            
                            for add1 in range(n):
                                if time.time() > deadline - 10.0:
                                    break
                                if add1 in working_set_set:
                                    continue
                                if not can_add(add1, temp_mask):
                                    continue
                                temp_mask2 = temp_mask | (1 << add1)
                                
                                for add2 in range(n):
                                    if time.time() > deadline - 10.0:
                                        break
                                    if add2 in working_set_set or add2 == add1:
                                        continue
                                    if not can_add(add2, temp_mask2):
                                        continue
                                    
                                    new_score = working_score - weights[remove_i] - weights[remove_j] + weights[add1] + weights[add2]
                                    if new_score > working_score + 1e-12:
                                        new_working_set = [x for x in working_set if x != remove_i and x != remove_j]
                                        new_working_set.append(add1)
                                        new_working_set.append(add2)
                                        working_set = new_working_set
                                        working_set_set = set(working_set)
                                        working_mask = 0
                                        for k in working_set:
                                            working_mask |= (1 << k)
                                        working_score = new_score
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                # Update global best if improved
                if working_score > best_score + 1e-12:
                    best_score = working_score
                    best_set_indices = list(working_set)
                    current_set = list(working_set)
    
    # Final high-precision polish: iterate 1-for-1 and 1-for-2 with tighter tolerance
    if time.time() < deadline - 5.0 and best_set_indices:
        working_set = list(best_set_indices)
        working_score = compute_score(working_set)
        improved = True
        polish_iter = 0
        max_polish = 50
        
        while improved and polish_iter < max_polish and time.time() < deadline - 5.0:
            improved = False
            polish_iter += 1
            working_set_set = set(working_set)
            working_mask = 0
            for i in working_set:
                working_mask |= (1 << i)
            
            # Try 1-for-1 swaps with very tight tolerance
            for remove_idx in range(len(working_set)):
                if time.time() > deadline - 5.0:
                    break
                remove_i = working_set[remove_idx]
                temp_mask = working_mask & ~(1 << remove_i)
                
                for add_i in range(n):
                    if add_i in working_set_set:
                        continue
                    if can_add(add_i, temp_mask):
                        new_score = working_score - weights[remove_i] + weights[add_i]
                        if new_score > working_score + 1e-15:
                            working_set[remove_idx] = add_i
                            working_set_set = set(working_set)
                            working_mask = 0
                            for k in working_set:
                                working_mask |= (1 << k)
                            working_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            # Try 1-for-2 swaps with very tight tolerance
            if not improved:
                for remove_idx in range(len(working_set)):
                    if time.time() > deadline - 5.0:
                        break
                    remove_i = working_set[remove_idx]
                    temp_mask = working_mask & ~(1 << remove_i)
                    
                    for add1 in range(n):
                        if time.time() > deadline - 5.0:
                            break
                        if add1 in working_set_set:
                            continue
                        if not can_add(add1, temp_mask):
                            continue
                        
                        temp_mask2 = temp_mask | (1 << add1)
                        for add2 in range(n):
                            if time.time() > deadline - 5.0:
                                break
                            if add2 in working_set_set or add2 == add1:
                                continue
                            if not can_add(add2, temp_mask2):
                                continue
                            
                            new_score = working_score - weights[remove_i] + weights[add1] + weights[add2]
                            if new_score > working_score + 1e-15:
                                working_set[remove_idx] = add1
                                working_set.append(add2)
                                working_set_set = set(working_set)
                                working_mask = 0
                                for k in working_set:
                                    working_mask |= (1 << k)
                                working_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Update global best if improved
            if working_score > best_score + 1e-15:
                best_score = working_score
                best_set_indices = list(working_set)
    
    # Convert indices back to values
    result = [nums[i] for i in best_set_indices]
    
    return result
# EVOLVE-BLOCK-END

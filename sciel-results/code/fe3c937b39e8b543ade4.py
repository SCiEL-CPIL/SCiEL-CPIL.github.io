"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    N = 512
    size = 511  # numbers 1..511
    start_time = time.time()
    
    # Weights: 1 / (512 - a)
    weights = [0.0] * size
    for i in range(size):
        a = i + 1
        weights[i] = 1.0 / (N - a)
        
    # Precompute conflict masks
    # conflict_mask[i] is an integer where bit j is set if gcd(i+1, j+1) > 1
    conflict_mask = [0] * size
    for i in range(size):
        a = i + 1
        mask = 0
        for j in range(size):
            if i == j:
                continue
            b = j + 1
            if gcd(a, b) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Helper to check compatibility
    def is_compatible(i, chosen_mask):
        return (conflict_mask[i] & chosen_mask) == 0
        
    # Greedy Initialization
    # Sort indices by weight descending
    indices_by_weight = sorted(range(size), key=lambda x: weights[x], reverse=True)
    
    best_chosen_mask = 0
    best_score = 0.0
    
    # Try a few different greedy strategies or just one strong one
    # Strategy 1: Pure Greedy
    chosen_mask = 0
    for i in indices_by_weight:
        if is_compatible(i, chosen_mask):
            chosen_mask |= (1 << i)
            
    # Calculate score
    score = 0.0
    for i in range(size):
        if (chosen_mask >> i) & 1:
            score += weights[i]
    best_score = score
    best_chosen_mask = chosen_mask
    
    # Local Search
    # We use a loop of Add and Best-Swap
    
    improved = True
    while improved and (time.time() - start_time) < 110:
        improved = False
        chosen_mask = best_chosen_mask
        
        # Phase 1: Add any compatible unchosen number
        for i in range(size):
            if not ((chosen_mask >> i) & 1):
                if is_compatible(i, chosen_mask):
                    chosen_mask |= (1 << i)
                    improved = True
        
        # Phase 2: Best Single Swap
        # Find the swap (remove i, add j) that gives the maximum gain
        max_gain = 0.0
        best_swap_i = -1
        best_swap_j = -1
        
        # Iterate over chosen i
        for i in range(size):
            if not ((chosen_mask >> i) & 1):
                continue
                
            # Remove i temporarily
            temp_mask = chosen_mask & ~(1 << i)
            
            # Find best j to add
            best_j_for_i = -1
            best_gain_for_i = 0.0
            
            for j in range(size):
                if j == i:
                    continue
                if (temp_mask >> j) & 1:
                    continue
                if is_compatible(j, temp_mask):
                    gain = weights[j] - weights[i]
                    if gain > best_gain_for_i:
                        best_gain_for_i = gain
                        best_j_for_i = j
                        
            if best_gain_for_i > max_gain:
                max_gain = best_gain_for_i
                best_swap_i = i
                best_swap_j = best_j_for_i
                
        if best_swap_i != -1 and max_gain > 1e-12:
            # Apply swap
            chosen_mask &= ~(1 << best_swap_i)
            chosen_mask |= (1 << best_swap_j)
            improved = True
            
        # Phase 3: 2-Swap Heuristic (Optimized)
        # Try to replace 2 lowest weight chosen with 2 best unchosen
        # This is expensive, so we limit it.
        # Get chosen indices
        chosen_indices = [i for i in range(size) if (chosen_mask >> i) & 1]
        unchosen_indices = [i for i in range(size) if not ((chosen_mask >> i) & 1)]
        
        # Sort chosen by weight ascending (worst first)
        chosen_indices.sort(key=lambda x: weights[x])
        # Sort unchosen by weight descending (best first)
        unchosen_indices.sort(key=lambda x: weights[x], reverse=True)
        
        # Only consider top K unchosen and bottom K chosen to limit search
        K = 10
        candidates_remove = chosen_indices[:K]
        candidates_add = unchosen_indices[:K]
        
        best_2_swap_gain = 0.0
        best_2_swap = None
        
        for i1 in candidates_remove:
            for i2 in candidates_remove:
                if i1 >= i2:
                    continue
                # Remove i1, i2
                temp_mask = chosen_mask & ~(1 << i1) & ~(1 << i2)
                base_weight = weights[i1] + weights[i2]
                
                # Find best pair j1, j2 from candidates_add
                # Must be compatible with temp_mask and each other
                best_pair_gain = 0.0
                best_pair = None
                
                for j1 in candidates_add:
                    if j1 == i1 or j1 == i2:
                        continue
                    if (temp_mask >> j1) & 1:
                        continue
                    if not is_compatible(j1, temp_mask):
                        continue
                    
                    for j2 in candidates_add:
                        if j2 <= j1:
                            continue
                        if j2 == i1 or j2 == i2:
                            continue
                        if (temp_mask >> j2) & 1:
                            continue
                        if not is_compatible(j2, temp_mask):
                            continue
                        
                        # Check compatibility between j1 and j2
                        if (conflict_mask[j1] >> j2) & 1:
                            continue
                        
                        gain = weights[j1] + weights[j2] - base_weight
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (i1, i2, j1, j2)
                            
                if best_pair and best_pair_gain > best_2_swap_gain:
                    best_2_swap_gain = best_pair_gain
                    best_2_swap = best_pair
                    
        if best_2_swap and best_2_swap_gain > 1e-12:
            i1, i2, j1, j2 = best_2_swap
            chosen_mask &= ~(1 << i1)
            chosen_mask &= ~(1 << i2)
            chosen_mask |= (1 << j1)
            chosen_mask |= (1 << j2)
            improved = True
            
        # Update best if improved
        current_score = 0.0
        for i in range(size):
            if (chosen_mask >> i) & 1:
                current_score += weights[i]
                
        if current_score > best_score:
            best_score = current_score
            best_chosen_mask = chosen_mask
        else:
            # If no improvement in this iteration, break? 
            # Since we did best swap and add, if no improvement, we are at a local optimum for this neighborhood.
            # We can break to save time or try random restarts.
            # Given time constraints, let's break if we don't improve.
            # Actually, the loop condition `improved` handles this.
            # If `improved` is False, the while loop will terminate.
            pass

    # Convert mask to list
    result = []
    for i in range(size):
        if (best_chosen_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END

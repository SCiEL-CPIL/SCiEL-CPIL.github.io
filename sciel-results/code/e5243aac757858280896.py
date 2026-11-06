"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)
    
    # Precompute intersection of every pair of subsets
    # pair_inter[i][j] = subsets[i] & subsets[j]
    # This is a 1820x1820 array, which is ~3.3M entries
    # We'll use a list of lists of ints
    
    def can_add_fast(mask_new, chosen_indices):
        """
        Check if mask_new can be added to the family defined by chosen_indices.
        For a sunflower with core I: need A, B in chosen such that
        A & mask_new == I, B & mask_new == I, and A & B == I.
        """
        # Group chosen sets by their intersection with mask_new
        # Only care about intersections that are subsets of mask_new
        # Since mask_new has 4 bits, intersection can be 0, 1, 2, or 3 bits
        groups = defaultdict(list)
        for idx in chosen_indices:
            m = all_subsets[idx]
            inter = m & mask_new
            groups[inter].append(idx)
        
        for inter, group in groups.items():
            L = len(group)
            if L < 2:
                continue
            # Check all pairs in this group
            for i in range(L):
                gi = group[i]
                for j in range(i + 1, L):
                    gj = group[j]
                    if (all_subsets[gi] & all_subsets[gj]) == inter:
                        return False
        return True

    best_family = []
    best_size = 0
    
    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen_indices = []
    for idx in range(num_subsets):
        if time.time() > deadline:
            break
        mask = all_subsets[idx]
        if can_add_fast(mask, chosen_indices):
            chosen_indices.append(idx)
            
    if len(chosen_indices) > best_size:
        best_size = len(chosen_indices)
        best_family = chosen_indices[:]

    # Strategy 2: Randomized Greedy Restarts
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:  # 3s safety margin
            break
        
        random.seed(12345 + restart)
        order = list(range(num_subsets))
        random.shuffle(order)
        
        chosen_indices = []
        for idx in order:
            if time.time() > deadline - 3.0:
                break
            mask = all_subsets[idx]
            if can_add_fast(mask, chosen_indices):
                chosen_indices.append(idx)
                
        if len(chosen_indices) > best_size:
            best_size = len(chosen_indices)
            best_family = chosen_indices[:]
            
            if best_size >= 250:
                break

    # Strategy 3: Local search - try to swap out a set for a better one
    # For the best family found, try local improvements
    if best_size > 0 and time.time() < deadline - 10.0:
        current_family = best_family[:]
        current_set = set(current_family)
        
        # Try multiple rounds of local search
        max_rounds = 5
        for round_num in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try removing each set and seeing if we can add a better one
            for i in range(len(current_family)):
                if time.time() > deadline - 5.0:
                    break
                
                removed_idx = current_family[i]
                remaining = [x for x in current_family if x != removed_idx]
                remaining_set = set(remaining)
                
                # Try to add sets not in the family
                for candidate_idx in range(num_subsets):
                    if candidate_idx in remaining_set:
                        continue
                    
                    mask = all_subsets[candidate_idx]
                    if can_add_fast(mask, remaining):
                        # Check if this candidate is "better" than removed
                        # For now, just accept any valid addition
                        new_family = remaining + [candidate_idx]
                        if len(new_family) > len(current_family):
                            current_family = new_family
                            current_set = set(current_family)
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]

    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_subsets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END

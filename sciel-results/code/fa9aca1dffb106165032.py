"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-sets form a sunflower (all three pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _check_valid(chosen, new_set):
    """Check if adding new_set to chosen creates a sunflower triple."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_sunflower_triple(chosen[i], chosen[j], new_set):
                return False
    return True

def _greedy_construct(order, deadline):
    """Greedy construction following a given order of subsets."""
    chosen = []
    chosen_sets = []
    for s in order:
        if time.time() > deadline:
            break
        if _check_valid(chosen_sets, s):
            chosen_sets.append(s)
            chosen.append(sorted(s))
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16]
    all_subsets = [set(row) for row in combinations(range(1, 17), 4)]
    
    # Precompute pairwise intersections for efficiency
    # For each pair (i,j), store the intersection set
    n = len(all_subsets)
    # We'll use a more efficient approach: for each new candidate, check against existing
    
    # Strategy: Multiple greedy passes with different orderings
    best_family = []
    
    # Pass 1: Natural order
    family1 = _greedy_construct(all_subsets, deadline)
    if len(family1) > len(best_family):
        best_family = family1
    
    if time.time() > deadline:
        return best_family
    
    # Pass 2: Random order
    random.seed(42)
    shuffled = all_subsets[:]
    random.shuffle(shuffled)
    family2 = _greedy_construct(shuffled, deadline)
    if len(family2) > len(best_family):
        best_family = family2
    
    if time.time() > deadline:
        return best_family
    
    # Pass 3: Another random order with different seed
    random.seed(123)
    shuffled2 = all_subsets[:]
    random.shuffle(shuffled2)
    family3 = _greedy_construct(shuffled2, deadline)
    if len(family3) > len(best_family):
        best_family = family3
    
    if time.time() > deadline:
        return best_family
    
    # Pass 4: Local improvement - try to swap out sets for better ones
    # Start from best_family and try to improve
    current = [set(s) for s in best_family]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected set and removing one that causes conflict
        for s in all_subsets:
            if time.time() > deadline:
                break
            if s in current:
                continue
            # Check if we can add s by removing one existing set
            for i, existing in enumerate(current):
                # Check if s conflicts with any pair in current \ {existing}
                valid = True
                for j in range(len(current)):
                    if j == i:
                        continue
                    for k in range(j + 1, len(current)):
                        if k == i:
                            continue
                        if _is_sunflower_triple(existing, current[j], current[k]):
                            # This pair already exists, skip
                            pass
                        if _is_sunflower_triple(s, current[j], current[k]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Try the swap
                    new_current = current[:i] + [s] + current[i+1:]
                    # Verify no sunflower in new_current
                    is_valid = True
                    for a in range(len(new_current)):
                        for b in range(a + 1, len(new_current)):
                            for c in range(b + 1, len(new_current)):
                                if _is_sunflower_triple(new_current[a], new_current[b], new_current[c]):
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    if is_valid:
                        current = new_current
                        improved = True
                        break
            if improved:
                break
    
    result = [sorted(s) for s in current]
    if len(result) > len(best_family):
        best_family = result
    
    return best_family
# EVOLVE-BLOCK-END

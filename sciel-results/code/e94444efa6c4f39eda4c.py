"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with a >= 1, a+3d <= n, d >= 1
    # For each element x, precompute which other elements it conflicts with in a 4-AP
    # Actually, for the can_add check, we need to check if adding x creates a 4-AP
    # x can be at position 0, 1, 2, or 3 in a 4-AP
    
    # Precompute for each x, the list of (d, other_elements) that would form a 4-AP
    # If x is at position i in a 4-AP (a, a+d, a+2d, a+3d):
    #   i=0: a=x, need x+d, x+2d, x+3d in set
    #   i=1: a+d=x, need a=x-d, x+2d, x+3d in set -> x-d, x+d, x+2d
    #   i=2: a+2d=x, need a=x-2d, x-d, x+d in set
    #   i=3: a+3d=x, need x-3d, x-2d, x-d in set
    
    # For speed, use a boolean array instead of set
    def can_add_fast(x, present):
        """present is a numpy boolean array of size n+1 (1-indexed)"""
        # x as 4th term (i=3): need x-d, x-2d, x-3d all present
        for d in range(1, x // 3 + 1):
            if present[x-d] and present[x-2*d] and present[x-3*d]:
                return False
        # x as 3rd term (i=2): need x-2d, x-d, x+d all present
        max_d = min(x - 2, n - x)
        for d in range(1, max_d + 1):
            if present[x-2*d] and present[x-d] and present[x+d]:
                return False
        # x as 2nd term (i=1): need x-d, x+d, x+2d all present
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if present[x-d] and present[x+d] and present[x+2*d]:
                return False
        # x as 1st term (i=0): need x+d, x+2d, x+3d all present
        for d in range(1, (n - x) // 3 + 1):
            if present[x+d] and present[x+2*d] and present[x+3*d]:
                return False
        return True
    
    def greedy_construct(order, present=None):
        """Greedy construction given an ordering of candidates."""
        if present is None:
            present = np.zeros(n + 1, dtype=bool)
        selected = set()
        for x in order:
            if not present[x] and can_add_fast(x, present):
                present[x] = True
                selected.add(x)
        return selected, present
    
    def local_search(selected, present, max_iters=200):
        """Try to improve selected by adding/removing elements."""
        current_size = len(selected)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding any element that can be added
            for x in range(1, n + 1):
                if not present[x] and can_add_fast(x, present):
                    present[x] = True
                    selected.add(x)
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try removing one element to allow adding two
                # Sample a few removals to save time
                candidates_to_remove = list(selected)
                random.shuffle(candidates_to_remove)
                found_improvement = False
                for x in candidates_to_remove[:20]:  # Try up to 20 removals
                    present[x] = False
                    trial_size = current_size - 1
                    # Try to add elements
                    added = 0
                    added_elements = []
                    for y in range(1, n + 1):
                        if not present[y] and can_add_fast(y, present):
                            present[y] = True
                            added_elements.append(y)
                            added += 1
                    if added > 1:
                        # Keep the improvement
                        for y in added_elements:
                            selected.add(y)
                        selected.discard(x)
                        current_size = trial_size + added
                        found_improvement = True
                        break
                    else:
                        # Revert
                        for y in added_elements:
                            present[y] = False
                            selected.discard(y)
                        present[x] = True
                        current_size = trial_size + 1
            
            if not improved and not found_improvement:
                break
        
        return selected, present
    
    # Phase 1: Multiple randomized greedy passes with local search
    best_set = set()
    best_present = None
    best_size = 0
    
    # Deterministic baseline: start with numbers that are 1 mod 3 or similar
    # Actually, let's just do greedy in natural order
    selected, present = greedy_construct(range(1, n + 1))
    selected, present = local_search(selected, present, max_iters=100)
    best_size = len(selected)
    best_set = set(selected)
    best_present = present.copy()
    
    # Randomized greedy with different orderings
    candidates = list(range(1, n + 1))
    iteration = 0
    while time.time() < deadline - 15:  # Reserve 15s for final polish
        random.shuffle(candidates)
        selected, present = greedy_construct(candidates)
        selected, present = local_search(selected, present, max_iters=50)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_present = present.copy()
        
        iteration += 1
    
    # Phase 2: More aggressive local search on best found
    if time.time() < deadline - 2:
        improved_selected, improved_present = local_search(best_set, best_present.copy(), max_iters=500)
        if len(improved_selected) > best_size:
            best_size = len(improved_selected)
            best_set = improved_selected
            best_present = improved_present
    
    return sorted(best_set)

# EVOLVE-BLOCK-END

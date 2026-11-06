"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from math import gcd

def construct():
    N = 512
    numbers = list(range(1, 512))
    
    # Precompute weights
    weights = {a: 1.0 / (N - a) for a in numbers}
    
    # Build conflict graph using bitsets for speed
    # conflict_mask[a] is a set of b such that gcd(a, b) > 1
    conflict_mask = {}
    for a in numbers:
        mask = set()
        for b in numbers:
            if a != b and gcd(a, b) > 1:
                mask.add(b)
        conflict_mask[a] = mask
    
    # Greedy construction: pick highest weight first, then next that's coprime to all chosen
    # Sort by weight descending
    sorted_by_weight = sorted(numbers, key=lambda x: -weights[x])
    
    chosen = []
    chosen_set = set()
    for a in sorted_by_weight:
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    # Local search: try to improve by swapping out a composite for multiple coprime numbers
    # Repeat until no improvement found
    best_score = sum(weights[a] for a in chosen)
    
    for iteration in range(200):
        improved = False
        best_swap = None
        best_gain = 0
        
        for a in chosen:
            # Try removing a and adding a set of numbers that are coprime to (chosen - {a})
            remaining = [b for b in chosen if b != a]
            remaining_set = set(remaining)
            
            # Candidates: all b not in chosen such that gcd(b, c) == 1 for all c in remaining
            candidates = []
            for b in numbers:
                if b in chosen_set:
                    continue
                if all(gcd(b, c) == 1 for c in remaining):
                    candidates.append(b)
            
            # Sort candidates by weight descending
            candidates.sort(key=lambda x: -weights[x])
            
            # Greedily build a set from candidates that are pairwise coprime and coprime to remaining
            new_set = []
            new_set_list = []
            for b in candidates:
                if all(gcd(b, c) == 1 for c in new_set_list) and all(gcd(b, c) == 1 for c in remaining):
                    new_set.append(b)
                    new_set_list.append(b)
            
            gain = sum(weights[b] for b in new_set_list) - weights[a]
            if gain > best_gain:
                best_gain = gain
                best_swap = (a, new_set_list)
        
        if best_swap is not None and best_gain > 1e-12:
            a_remove, a_add = best_swap
            chosen = [b for b in chosen if b != a_remove] + a_add
            chosen_set = set(chosen)
            best_score = sum(weights[a] for a in chosen)
            improved = True
        
        if not improved:
            break
    
    return chosen
# EVOLVE-BLOCK-END

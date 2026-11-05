"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-element sets form a sunflower.
    A sunflower (including empty core) means all three pairwise intersections are equal.
    """
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    n = 16
    all_quads = list(combinations(range(1, n + 1), 4))
    all_sets = [set(q) for q in all_quads]
    
    # Greedy construction with randomized order and multiple restarts
    best_family = []
    best_size = 0
    
    # Precompute all pairs for faster checking? Or just check on the fly.
    # With max 250 sets, pairs are ~31k. Checking each new set against ~31k pairs is fast.
    
    num_restarts = 200
    random.seed(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize order of all quads
        indices = list(range(len(all_quads)))
        random.shuffle(indices)
        
        current_family = []
        current_sets = []
        
        for idx in indices:
            s = all_sets[idx]
            valid = True
            # Check against all existing pairs in current_family
            # Optimization: if family is large, this loop is the bottleneck.
            # We can break early if we find a violation.
            for i in range(len(current_sets)):
                a = current_sets[i]
                for j in range(i + 1, len(current_sets)):
                    b = current_sets[j]
                    if _is_sunflower_triple(a, b, s):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_family.append(idx)
                current_sets.append(s)
                
            if len(current_sets) > best_size:
                best_size = len(current_family)
                best_family = current_family[:]
                
            # Optional: early stop if we hit the theoretical max or some high threshold?
            # The max is 250. If we reach 250, we can't do better.
            if best_size >= 250:
                break
                
        if best_size >= 250:
            break

    # Convert back to sorted lists
    result = [sorted(all_sets[i]) for i in best_family]
    return result
# EVOLVE-BLOCK-END

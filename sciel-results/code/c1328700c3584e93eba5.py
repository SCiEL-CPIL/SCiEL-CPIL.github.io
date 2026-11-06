"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _to_mask(s):
    mask = 0
    for x in s:
        mask |= (1 << (x - 1))
    return mask

def _from_mask(mask):
    return [i + 1 for i in range(16) if (mask >> i) & 1]

def _greedy_build(order, deadline):
    """
    order: list of integer masks
    Returns list of integer masks
    """
    chosen = []
    n_chosen = 0
    for s in order:
        if time.time() > deadline:
            break
        ok = True
        # Check against all pairs in chosen
        for i in range(n_chosen):
            a = chosen[i]
            for j in range(i + 1, n_chosen):
                b = chosen[j]
                # Check if a, b, s form a sunflower
                # Intersections: a&b, a&s, b&s must be equal
                ab = a & b
                as_ = a & s
                bs = b & s
                if ab == as_ and ab == bs:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(s)
            n_chosen += 1
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Generate all 4-element subsets of [1, 16] as masks
    all_masks = []
    for c in combinations(range(1, 17), 4):
        all_masks.append(_to_mask(c))
    
    all_sets_list = [list(c) for c in combinations(range(1, 17), 4)]
    # Map mask to original set for output
    mask_to_set = {}
    for m, s in zip(all_masks, all_sets_list):
        mask_to_set[m] = s

    rng = random.Random(42)
    best_family = []
    
    # Run multiple greedy passes with different orderings
    # Limit the number of passes to ensure we don't timeout
    max_passes = 20
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        order = list(all_masks)
        rng.shuffle(order)
        
        # Use a slightly earlier deadline for the greedy build itself to allow for processing
        fam = _greedy_build(order, deadline - 1.0)
        
        if len(fam) > len(best_family):
            best_family = fam
    
    # Convert best family masks back to sorted lists
    result = [mask_to_set[m] for m in best_family]
    
    return result
# EVOLVE-BLOCK-END

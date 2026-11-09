"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1,16] as bitmasks
# Use 0-indexed for bitmask efficiency, convert back to 1-indexed at the end
_ALL_SUBSETS = list(combinations(range(16), 4))
_ALL_BITMASKS = [0] * len(_ALL_SUBSETS)
for i, sub in enumerate(_ALL_SUBSETS):
    mask = 0
    for x in sub:
        mask |= (1 << x)
    _ALL_BITMASKS[i] = mask

def _check_sunflower(a, b, c):
    """Check if three 4-element sets (as bitmasks) form a sunflower.
    A sunflower means all three pairwise intersections are equal.
    For 4-sets, the common intersection can have size 0, 1, 2, 3, or 4.
    Size 4 means a==b==c which can't happen for distinct sets.
    """
    ab = a & b
    ac = a & c
    bc = b & c
    # All three pairwise intersections must be equal
    return ab == ac == bc

def _is_valid_addition(candidate_mask, chosen_masks):
    """Check if adding candidate_mask to chosen_masks creates no sunflower."""
    for i in range(len(chosen_masks)):
        a = chosen_masks[i]
        for j in range(i + 1, len(chosen_masks)):
            b = chosen_masks[j]
            if _check_sunflower(a, b, candidate_mask):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Greedy with multiple random orderings to find larger families
    best_size = 0
    best_family = []
    
    # First try natural order (same as seed)
    order = list(range(len(_ALL_SUBSETS)))
    chosen = []
    for idx in order:
        mask = _ALL_BITMASKS[idx]
        if _is_valid_addition(mask, chosen):
            chosen.append(mask)
    
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = list(chosen)
    
    # Try random restarts with different orderings
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = list(range(len(_ALL_SUBSETS)))
        rng.shuffle(order)
        chosen = []
        for idx in order:
            mask = _ALL_BITMASKS[idx]
            if _is_valid_addition(mask, chosen):
                chosen.append(mask)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
        
        # If we've reached a good size, try local improvement
        if best_size >= 100:
            # Try to add more sets by checking all unchosen sets
            chosen_set = set(best_family)
            for idx in range(len(_ALL_SUBSETS)):
                mask = _ALL_BITMASKS[idx]
                if mask not in chosen_set:
                    if _is_valid_addition(mask, best_family):
                        best_family.append(mask)
                        chosen_set.add(mask)
                        best_size += 1
            break
    
    # Convert bitmasks back to 1-indexed sorted lists
    result = []
    for mask in best_family:
        elems = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END

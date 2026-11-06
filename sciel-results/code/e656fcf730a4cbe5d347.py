import random
import time
from itertools import combinations

# Precompute all 4-subsets of [1, 16] as bitmasks
# Elements are 1..16, map to bits 0..15
ALL_SUBSETS_MASKS = []
for c in combinations(range(16), 4):
    mask = 0
    for x in c:
        mask |= (1 << x)
    ALL_SUBSETS_MASKS.append(mask)

def can_add(family_masks, new_mask):
    """
    family_masks: list of int masks
    new_mask: int mask
    """
    n = len(family_masks)
    if n == 0:
        return True
    
    # Precompute intersection of new_mask with all sets in family
    inters = [new_mask & m for m in family_masks]
    
    # Check all pairs
    for i in range(n):
        mi = family_masks[i]
        inter_i = inters[i]
        for j in range(i):
            if inter_i == inters[j] == (mi & family_masks[j]):
                return False
    return True

def construct():
    best_family = []
    deadline = time.time() + 85
    
    while time.time() < deadline:
        current_family = []
        candidates = ALL_SUBSETS_MASKS[:]
        random.shuffle(candidates)
        
        for m in candidates:
            if can_add(current_family, m):
                current_family.append(m)
                
        if len(current_family) > len(best_family):
            best_family = current_family
            if len(best_family) == 250:
                break
                
    # Convert masks back to sorted lists of 1-based indices
    result = []
    for m in best_family:
        elems = [i + 1 for i in range(16) if (m >> i) & 1]
        result.append(sorted(elems))
    return result

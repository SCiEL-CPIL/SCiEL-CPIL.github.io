"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import random

def _precompute_sets():
    """Precompute all 4-subsets, indices, bitmasks, and mask-to-index map."""
    all_sets = []
    set_to_idx = {}
    bitmasks = []
    mask_to_idx = {}
    
    for c in combinations(range(1, 17), 4):
        s = frozenset(c)
        idx = len(all_sets)
        all_sets.append(s)
        set_to_idx[s] = idx
        
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        bitmasks.append(mask)
        mask_to_idx[mask] = idx
        
    return all_sets, set_to_idx, bitmasks, mask_to_idx

def construct():
    all_sets, set_to_idx, bitmasks, mask_to_idx = _precompute_sets()
    n_sets = len(all_sets)
    universe_mask = (1 << 16) - 1
    
    best_size = 0
    best_chosen = []
    
    rng = random.Random(42)
    
    # Helper to get elements from a mask
    def mask_to_list(mask):
        return [i + 1 for i in range(16) if (mask >> i) & 1]

    # Use more trials as the algorithm is faster
    for trial in range(300):
        order = list(range(n_sets))
        rng.shuffle(order)
        
        chosen = []
        chosen_masks = []
        banned = set()
        
        for idx in order:
            if idx in banned:
                continue
            
            s_mask = bitmasks[idx]
            
            chosen.append(idx)
            chosen_masks.append(s_mask)
            
            # Update banned set for pairs (s, prev_s)
            for i in range(len(chosen_masks) - 1):
                prev_mask = chosen_masks[i]
                
                intersection_mask = s_mask & prev_mask
                union_mask = s_mask | prev_mask
                r_mask = universe_mask & ~union_mask
                
                k = 4 - intersection_mask.bit_count()
                
                if 0 <= k <= 16:
                    r_elements = mask_to_list(r_mask)
                    if len(r_elements) >= k:
                        for c_prime in combinations(r_elements, k):
                            # Construct candidate mask
                            cand_mask = intersection_mask
                            for x in c_prime:
                                cand_mask |= (1 << (x - 1))
                            
                            # Check if this mask corresponds to a valid set
                            if cand_mask in mask_to_idx:
                                banned.add(mask_to_idx[cand_mask])
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
    
    return [sorted(all_sets[i]) for i in best_chosen]
# EVOLVE-BLOCK-END

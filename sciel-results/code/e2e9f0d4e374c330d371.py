"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute all 4-element subsets as bitmasks for fast intersection checks
    all_masks = []
    all_combos = []
    for combo in itertools.combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        all_combos.append(combo)
    
    n = len(all_masks)
    
    def mask_to_list(mask):
        return [i + 1 for i in range(16) if (mask >> i) & 1]
    
    def can_add(family_masks, new_mask):
        """Check if new_mask can be added to family_masks without creating a sunflower triple."""
        for i in range(len(family_masks)):
            for j in range(i + 1, len(family_masks)):
                a = family_masks[i]
                b = family_masks[j]
                c = new_mask
                # Check if a & b == a & c == b & c
                ab = a & b
                ac = a & c
                bc = b & c
                if ab == ac == bc:
                    return False
        return True
    
    def greedy_pass(order):
        """Greedy pass with given order of masks."""
        family = []
        for mask in order:
            if can_add(family, mask):
                family.append(mask)
        return family
    
    # Phase 1: Try multiple deterministic orderings
    best_family = []
    best_size = 0
    
    # Natural lexicographic order
    if time.time() < deadline:
        fam = greedy_pass(all_masks)
        if len(fam) > best_size:
            best_family = fam
            best_size = len(fam)
    
    # Reverse lexicographic order
    if time.time() < deadline:
        fam = greedy_pass(all_masks[::-1])
        if len(fam) > best_size:
            best_family = fam
            best_size = len(fam)
    
    # A few random orderings
    random.seed(42)
    for _ in range(3):
        if time.time() >= deadline - 5:
            break
        order = all_masks[:]
        random.shuffle(order)
        fam = greedy_pass(order)
        if len(fam) > best_size:
            best_family = fam
            best_size = len(fam)
    
    # Phase 2: Local search to try to improve
    if best_family and time.time() < deadline - 3:
        family = list(best_family)
        in_family = set(family)
        candidates = [m for m in all_masks if m not in in_family]
        random.shuffle(candidates)
        
        no_improve = 0
        max_no_improve = 200
        
        while candidates and no_improve < max_no_improve and time.time() < deadline - 1:
            # Pick a random candidate
            idx = random.randint(0, len(candidates) - 1)
            c = candidates[idx]
            
            if can_add(family, c):
                family.append(c)
                in_family.add(c)
                candidates.pop(idx)
                no_improve = 0
            else:
                no_improve += 1
                # Occasionally try a swap
                if no_improve % 20 == 0 and family:
                    rand_idx = random.randint(0, len(family) - 1)
                    old = family[rand_idx]
                    # Check if c can be added if we remove old
                    temp_family = family[:rand_idx] + family[rand_idx + 1:]
                    if can_add(temp_family, c):
                        family = temp_family
                        family.append(c)
                        in_family.discard(old)
                        in_family.add(c)
                        candidates.pop(idx)
                        candidates.append(old)
                        no_improve = 0
        
        if len(family) > best_size:
            best_family = family
            best_size = len(family)
    
    # Convert to sorted lists for output
    result = [mask_to_list(m) for m in best_family]
    return result
# EVOLVE-BLOCK-END

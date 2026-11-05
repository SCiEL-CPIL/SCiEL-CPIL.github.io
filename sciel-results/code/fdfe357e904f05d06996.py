# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 60s limit (or 90s)
    
    # All 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    
    # Convert to masks
    masks = []
    mask_to_set = {}
    for s in all_sets:
        m = 0
        for x in s:
            m |= 1 << (x - 1)
        masks.append(m)
        mask_to_set[m] = s
        
    # Precompute bit counts? Not strictly needed for the logic, but might help.
    
    best_size = 0
    best_masks = []
    
    random.seed(42)
    num_restarts = 10000 # High number, will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        shuffled_masks = masks[:]
        random.shuffle(shuffled_masks)
        
        family = []
        family_masks = [] # Keep as list of ints for fast access
        
        for m in shuffled_masks:
            # Check if m can be added to family
            is_valid = True
            n = len(family_masks)
            
            if n >= 2:
                # Compute intersections of m with all family members
                # I_i = family_masks[i] & m
                intersections = [fm & m for fm in family_masks]
                
                # Group indices by intersection value
                groups = defaultdict(list)
                for i, inter in enumerate(intersections):
                    groups[inter].append(i)
                    
                # Check each group for conflicts
                for I, indices in groups.items():
                    if len(indices) < 2:
                        continue
                    
                    # We need to check if there exist i, j in indices
                    # such that (family_masks[i] \ I) & (family_masks[j] \ I) == 0
                    # Let T_k = family_masks[k] & ~I
                    # Note: ~I in Python is infinite bits, so we must mask with 0xFFFF
                    # But family masks are only 16 bits.
                    # T_k = family_masks[k] & (~I & 0xFFFF)
                    # Or simply: T_k = family_masks[k] ^ I  (since I is subset of family_masks[k])
                    # Wait, I = family_masks[k] & m. So I is subset of family_masks[k].
                    # So T_k = family_masks[k] ^ I is correct.
                    
                    tails = []
                    for idx in indices:
                        fm = family_masks[idx]
                        # I is subset of fm, so fm ^ I gives the part of fm not in I
                        tail = fm ^ I
                        tails.append(tail)
                        
                    # Check if any two tails are disjoint
                    # L = len(tails)
                    # If L is small, O(L^2) is fine.
                    # Max L is N (250). 250^2 = 62500.
                    # This is per group. Sum of L^2 over groups is <= N^2.
                    # But usually groups are small.
                    
                    # Optimization: If any tail is 0, it's disjoint with everything.
                    # If any tail is small, check against others.
                    
                    conflict_found = False
                    for k in range(len(tails)):
                        tk = tails[k]
                        if tk == 0:
                            # Disjoint with any other tail?
                            # 0 & x == 0. Yes.
                            # But we need two DISTINCT sets.
                            # If there is at least one other set in the group, we have a conflict.
                            # Wait, the condition is (A\I) & (B\I) == 0.
                            # If A\I is empty, then A subset I. But A is size 4.
                            # If I is size 4, then A=I.
                            # If two sets A, B both equal I, they are not distinct?
                            # The problem says "three DISTINCT sets".
                            # So A, B, C must be distinct.
                            # In our check, A and B are from family, C is m.
                            # A and B are distinct indices.
                            # So if we find two distinct indices k, l with disjoint tails, it's a conflict.
                            
                            if len(tails) > 1:
                                conflict_found = True
                                break
                            
                        for l in range(k + 1, len(tails)):
                            tl = tails[l]
                            if (tk & tl) == 0:
                                conflict_found = True
                                break
                        if conflict_found:
                            break
                            
                    if conflict_found:
                        is_valid = False
                        break
            
            if is_valid:
                family.append(m)
                family_masks.append(m)
        
        if len(family) > best_size:
            best_size = len(family)
            best_masks = family[:]
            
        if best_size >= 250:
            break

    # Convert best_masks back to sorted lists
    result = []
    for m in best_masks:
        s = mask_to_set[m]
        result.append(sorted(s))
    
    return result

# EVOLVE-BLOCK-END

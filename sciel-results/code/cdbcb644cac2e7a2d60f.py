import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Generate all 4-subsets of [16] as bitmasks
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    n = len(all_masks)
    
    # Pre-compute mask to elements for output
    mask_to_elems = {m: tuple(i+1 for i in range(16) if (m >> i) & 1) for m in all_masks}
    
    best_family = []
    best_size = 0
    
    # Number of restarts to try
    # We don't know how long each restart takes, so we loop until deadline
    restart = 0
    while time.time() < deadline:
        restart += 1
        order = list(range(n))
        random.shuffle(order)
        
        family = []
        fam_set = set()
        
        # For each candidate S in shuffled order
        for idx in order:
            if time.time() > deadline:
                break
            
            S = all_masks[idx]
            
            # Check if S can be added to family
            # Group existing family members by their intersection with S
            groups = {}
            for a_idx in family:
                A = all_masks[a_idx]
                inter = A & S
                if inter not in groups:
                    groups[inter] = []
                groups[inter].append(a_idx)
            
            conflict = False
            for inter, members in groups.items():
                # Check if any pair in members has intersection == inter
                # Note: A & B == inter implies inter is subset of A and B.
                # Since inter = A & S, inter is subset of A.
                # So we just check A & B == inter.
                k = len(members)
                if k < 2:
                    continue
                # Optimization: if k is small, check all pairs
                for i in range(k):
                    A = all_masks[members[i]]
                    for j in range(i+1, k):
                        B = all_masks[members[j]]
                        if (A & B) == inter:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(idx)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # If we are getting consistently small families, maybe break early?
        # But let's try to finish the current restart if possible.
        # If time is tight, break outer loop.
        
    return [list(mask_to_elems[all_masks[i]]) for i in best_family]

import time
import random
from itertools import combinations

def _is_sunflower(a, b, c):
    """Check if three 4-element sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 16
    k = 4
    
    # All possible 4-element subsets of [0, 16) as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    total_sets = len(all_masks)
    
    # Pre-compute mask to elements for output
    mask_to_elems = {}
    for m in all_masks:
        elems = []
        for i in range(n):
            if m & (1 << i):
                elems.append(i + 1)
        mask_to_elems[m] = elems
    
    best_size = 0
    best_family = []
    
    def greedy_construct(order):
        """Build a family greedily given an order of candidate sets."""
        family = []
        # For speed, we check conflicts against all pairs in family
        for idx in order:
            if time.time() > deadline:
                break
            s = all_masks[idx]
            valid = True
            # Check against all pairs in family
            for i in range(len(family)):
                a = family[i]
                for j in range(i + 1, len(family)):
                    b = family[j]
                    if _is_sunflower(a, b, s):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                family.append(s)
        return family
    
    def local_search(family, max_time):
        """Improve family via swap moves: remove one, try to add a different one."""
        family = family[:]
        fam_set = set(family)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(len(family)):
                if time.time() > max_time:
                    break
                removed = family[i]
                family_without = family[:i] + family[i+1:]
                # Try to find a replacement
                for j in range(total_sets):
                    if time.time() > max_time:
                        break
                    s = all_masks[j]
                    if s in fam_set and s != removed:
                        continue
                    # Check if s can be added to family_without
                    valid = True
                    for a_idx in range(len(family_without)):
                        a = family_without[a_idx]
                        for b_idx in range(a_idx + 1, len(family_without)):
                            b = family_without[b_idx]
                            if _is_sunflower(a, b, s):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        family[i] = s
                        fam_set.remove(removed)
                        fam_set.add(s)
                        improved = True
                        break
                if improved:
                    break
        return family
    
    # Phase 1: Multiple greedy restarts with different random orders
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        order = list(range(total_sets))
        random.shuffle(order)
        family = greedy_construct(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # Local search on this family
        if len(family) >= best_size:
            remaining = deadline - 15.0
            if time.time() < remaining:
                family_ls = local_search(family, remaining)
                if len(family_ls) > best_size:
                    best_size = len(family_ls)
                    best_family = family_ls[:]
    
    # Phase 2: Try to add more sets to best_family
    if time.time() < deadline - 5.0:
        fam_set = set(best_family)
        for idx in range(total_sets):
            if time.time() > deadline - 5.0:
                break
            s = all_masks[idx]
            if s in fam_set:
                continue
            valid = True
            for i in range(len(best_family)):
                a = best_family[i]
                for j in range(i + 1, len(best_family)):
                    b = best_family[j]
                    if _is_sunflower(a, b, s):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_family.append(s)
                fam_set.add(s)
                best_size = len(best_family)
    
    # Convert to output format
    result = [mask_to_elems[m] for m in best_family]
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _to_mask(s):
    """Convert a set of elements 1..16 to a bitmask."""
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_set(m):
    """Convert bitmask back to sorted list."""
    return [i+1 for i in range(16) if m & (1 << i)]

def _is_sunflower_masks(m1, m2, m3):
    """Check if three bitmask sets form a sunflower."""
    return (m1 & m2) == (m1 & m3) == (m2 & m3)

def _greedy_construct_masks(order_masks, deadline):
    """Greedy construction using bitmasks for speed."""
    chosen = []
    chosen_masks = []
    for m in order_masks:
        if time.time() > deadline:
            break
        conflict = False
        for i in range(len(chosen_masks)):
            mi = chosen_masks[i]
            for j in range(i+1, len(chosen_masks)):
                mj = chosen_masks[j]
                if _is_sunflower_masks(mi, mj, m):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen.append(m)
            chosen_masks.append(m)
    return chosen

def _local_search_masks(family_masks, all_masks, deadline):
    """Local search: try to add sets, then try swaps."""
    family_set = set(family_masks)
    family_list = list(family_masks)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add a set
        for m in all_masks:
            if m in family_set:
                continue
            conflict = False
            for i in range(len(family_list)):
                mi = family_list[i]
                for j in range(i+1, len(family_list)):
                    mj = family_list[j]
                    if _is_sunflower_masks(mi, mj, m):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                family_list.append(m)
                family_set.add(m)
                improved = True
                break
        # Try swaps: remove one, add another
        if not improved and time.time() < deadline:
            swapped = False
            for ri in range(len(family_list)):
                if time.time() > deadline:
                    break
                rm = family_list[ri]
                # Temporarily remove
                family_list_temp = family_list[:ri] + family_list[ri+1:]
                for m in all_masks:
                    if m in family_set:
                        continue
                    # Check if m can be added to family_list_temp
                    conflict = False
                    for i in range(len(family_list_temp)):
                        mi = family_list_temp[i]
                        for j in range(i+1, len(family_list_temp)):
                            mj = family_list_temp[j]
                            if _is_sunflower_masks(mi, mj, m):
                                conflict = True
                                break
                        if conflict:
                            break
                    if not conflict:
                        family_list[ri] = m
                        family_set.discard(rm)
                        family_set.add(m)
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
    return family_list

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Precompute all 4-element subsets as bitmasks
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    all_masks = [_to_mask(s) for s in all_sets]
    n_sets = len(all_masks)
    
    best_family = []
    
    # First: lexicographic order (baseline)
    if time.time() < deadline:
        family = _greedy_construct_masks(all_masks, deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Then: multiple random orderings with different seeds
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 5:  # reserve 5s for local search
        attempt += 1
        shuffled = all_masks[:]
        rng.shuffle(shuffled)
        family = _greedy_construct_masks(shuffled, deadline - 5)
        if len(family) > len(best_family):
            best_family = family
            if len(best_family) >= 250:
                break
    
    # Try local search to improve the best family
    if time.time() < deadline and len(best_family) < 250:
        improved_family = _local_search_masks(best_family, all_masks, deadline)
        if len(improved_family) > len(best_family):
            best_family = improved_family
    
    return [_mask_to_set(m) for m in best_family]
# EVOLVE-BLOCK-END

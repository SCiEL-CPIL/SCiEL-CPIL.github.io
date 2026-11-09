import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        
    total = len(all_masks)
    
    # Precompute intersections for all pairs
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = all_masks[i]
        for j in range(i + 1, total):
            val = mi & all_masks[j]
            inter[i][j] = val
            inter[j][i] = val
    
    def is_compatible(cand_idx, family):
        """Check if cand_idx can be added to family without creating a sunflower."""
        for i in range(len(family)):
            a_idx = family[i]
            ac = inter[a_idx][cand_idx]
            for j in range(i + 1, len(family)):
                b_idx = family[j]
                ab = inter[a_idx][b_idx]
                bc = inter[b_idx][cand_idx]
                if ab == ac == bc:
                    return False
        return True
    
    def run_greedy(order_indices):
        family = []
        for idx in order_indices:
            if time.time() > deadline - 2:
                break
            if is_compatible(idx, family):
                family.append(idx)
        return family
    
    def local_search(family, max_rounds=30):
        """Local search: try to add compatible sets or perform swaps to grow the family."""
        family = list(family)
        family_set = set(family)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            
            # Try to add any set not in family
            added = False
            candidates = [idx for idx in range(total) if idx not in family_set]
            random.shuffle(candidates)
            
            for cand in candidates:
                if time.time() > deadline - 3:
                    break
                if is_compatible(cand, family):
                    family.append(cand)
                    family_set.add(cand)
                    added = True
                    break
            
            if not added:
                # Try a swap: remove one, see if we can add two
                if len(family) >= 2:
                    swap_done = False
                    for i in range(len(family)):
                        if time.time() > deadline - 3:
                            break
                        removed = family[i]
                        temp_family = family[:i] + family[i+1:]
                        temp_set = family_set - {removed}
                        
                        # Try to find two sets that can be added
                        found = []
                        candidates2 = [idx for idx in range(total) if idx not in temp_set and idx != removed]
                        random.shuffle(candidates2)
                        
                        for cand in candidates2:
                            if len(found) >= 2:
                                break
                            if time.time() > deadline - 3:
                                break
                            if is_compatible(cand, temp_family + found):
                                found.append(cand)
                        
                        if len(found) >= 2:
                            family = temp_family + found
                            family_set = set(family)
                            swap_done = True
                            break
                    
                    if not swap_done:
                        break
        
        return family
    
    best_family = []
    
    # 1. Standard Lexicographic Order
    order = list(range(total))
    fam = run_greedy(order)
    if len(fam) > len(best_family):
        best_family = fam[:]
        
    # 2. Reverse Lexicographic Order
    if time.time() < deadline - 5:
        order_rev = list(reversed(range(total)))
        fam = run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 3. Random Permutations of the Universe
    for attempt in range(25):
        if time.time() > deadline - 8:
            break
        
        random.seed(42 + attempt)
        perm = list(range(n))
        random.shuffle(perm)
        
        keys = []
        for idx in range(total):
            mask = all_masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            elems_sorted_by_perm = sorted(elems)
            keys.append((tuple(elems_sorted_by_perm), idx))
            
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]
            
        # Local search on this family
        if len(fam) >= len(best_family) - 3:
            improved = local_search(fam, max_rounds=15)
            if len(improved) > len(best_family):
                best_family = improved[:]

    # 4. Pure random orderings
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        
        random.seed(100 + attempt)
        order_rand = list(range(total))
        random.shuffle(order_rand)
        
        fam = run_greedy(order_rand)
        if len(fam) > len(best_family):
            best_family = fam[:]
        
        if len(fam) >= len(best_family) - 3:
            improved = local_search(fam, max_rounds=10)
            if len(improved) > len(best_family):
                best_family = improved[:]

    # Final local search on best family
    if time.time() < deadline - 3 and len(best_family) > 0:
        improved = local_search(best_family, max_rounds=40)
        if len(improved) > len(best_family):
            best_family = improved[:]

    # Convert best_family indices back to lists of 1-based integers
    result = []
    for idx in best_family:
        elements = []
        for i in range(n):
            if all_masks[idx] & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result

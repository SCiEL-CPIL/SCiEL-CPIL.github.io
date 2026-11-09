import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total_masks = len(all_masks)
    
    def is_compatible(family, C):
        """Check if C can be added to family without forming a sunflower."""
        n_fam = len(family)
        if n_fam < 2:
            return True
        
        # Bucket by intersection with C
        buckets = {}
        for A in family:
            I = A & C
            if I not in buckets:
                buckets[I] = []
            buckets[I].append(A)
        
        for I, bl in buckets.items():
            if len(bl) < 2:
                continue
            for ia in range(len(bl)):
                A = bl[ia]
                for ib in range(ia + 1, len(bl)):
                    if (A & bl[ib]) == I:
                        return False
        return True
    
    best_family = []
    best_size = 0
    
    # Phase 1: Greedy with varied orderings
    for attempt in range(300):
        if time.time() > deadline - 15:
            break
        
        if attempt % 3 == 0:
            order = all_masks[:]
            random.shuffle(order)
        elif attempt % 3 == 1:
            # Sort by bitmask (deterministic base, then shuffle within groups)
            order = sorted(all_masks)
            random.shuffle(order)
        else:
            # Weighted random: prefer sets with lower "density" of common elements
            rand_ref = random.choice(all_masks)
            order = sorted(all_masks, key=lambda m: bin(m & rand_ref).count('1') + random.random())
        
        family = []
        for C in order:
            if time.time() > deadline - 15:
                break
            if is_compatible(family, C):
                family.append(C)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            if best_size >= 250:
                break
    
    # Phase 2: Local search - try swaps to improve
    if time.time() < deadline - 5 and best_size > 0:
        family_set = set(best_family)
        
        for ls_round in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try removing each element and finding a better replacement
            for i in range(len(best_family)):
                if time.time() > deadline - 5:
                    break
                
                removed = best_family[i]
                temp = best_family[:i] + best_family[i+1:]
                temp_set = family_set - {removed}
                
                # Try a few random candidates
                candidates = [m for m in all_masks if m not in temp_set]
                random.shuffle(candidates)
                
                found = False
                for C in candidates[:100]:
                    if is_compatible(temp, C):
                        new_fam = temp + [C]
                        if len(new_fam) > len(best_family):
                            best_family = new_fam
                            best_size = len(best_family)
                            family_set = set(best_family)
                            improved = True
                        found = True
                        break
                
                if found and improved:
                    break
            
            if not improved:
                break
    
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
    
    return result

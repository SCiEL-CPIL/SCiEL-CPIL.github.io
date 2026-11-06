import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit (evaluator kills at 90s but we target 60s)
    
    # Generate all 4-element subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)
    
    # 1. Initialize with a strong deterministic greedy (similar to seed)
    family = []
    for m in all_masks:
        valid = True
        # Check against existing family
        for i in range(len(family)):
            a = family[i]
            a_new = a & m
            for j in range(i + 1, len(family)):
                b = family[j]
                b_new = b & m
                if a_new == b_new and (a & b) == a_new:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            family.append(m)
            
    best_family = list(family)
    best_size = len(family)
    
    # 2. Local Search with Swaps
    # We use a simple validity check but limit the search space for swaps
    def is_valid_addition(fam, new_mask):
        for i in range(len(fam)):
            a = fam[i]
            a_new = a & new_mask
            for j in range(i + 1, len(fam)):
                b = fam[j]
                b_new = b & new_mask
                if a_new == b_new and (a & b) == a_new:
                    return False
        return True

    family_set = set(family)
    
    # Try to add any remaining sets
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for m in all_masks:
            if m in family_set:
                continue
            if is_valid_addition(family, m):
                family.append(m)
                family_set.add(m)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Iterated Local Search: Swap moves
    # To speed up, we pick a random set to remove and a random set to add
    current_family = list(family)
    current_set = set(family)
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline - 2:
            break
        
        # Pick a random set to remove
        if not current_family:
            break
        remove_idx = rng.randrange(len(current_family))
        removed_mask = current_family[remove_idx]
        
        # Temporarily remove
        current_family.pop(remove_idx)
        current_set.discard(removed_mask)
        
        # Try to add a random set from the pool
        # We try a few random candidates
        added = False
        candidates = rng.sample(all_masks, min(100, n_total))
        for new_mask in candidates:
            if new_mask in current_set:
                continue
            if is_valid_addition(current_family, new_mask):
                current_family.append(new_mask)
                current_set.add(new_mask)
                added = True
                break
        
        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            # Restore original
            current_family.insert(remove_idx, removed_mask)
            current_set.add(removed_mask)
            
        # If we hit the max, break
        if best_size >= 250:
            break

    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    
    return result

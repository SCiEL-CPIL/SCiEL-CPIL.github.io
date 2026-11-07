import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    def is_comparable(a, b):
        """Check if a and b are comparable (one is subset of other, strict)."""
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def count_comparables(family_masks):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        L = len(family_masks)
        for i in range(L):
            ai = family_masks[i]
            for j in range(i + 1, L):
                bj = family_masks[j]
                if (ai & ~bj) == 0 or (bj & ~ai) == 0:
                    cnt += 1
        return cnt

    # 1. Initialize with the longest chain (length 13)
    family = []
    cur = 0
    family.append(0)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    used = set(family)
    all_masks = list(range(total_masks))
    
    # 2. Greedily add sets that maximize comparable pairs with current family
    while len(family) < m:
        best_gain = -1
        best_mask = None
        
        # Optimization: We can precompute comparability with the current family for all masks?
        # No, family changes. But we can just loop.
        # 4096 masks * 40 family members = 160k ops per step.
        # 27 steps = 4.3M ops. Very fast.
        
        for s in all_masks:
            if s in used:
                continue
            
            gain = 0
            for f in family:
                if (s & ~f) == 0 or (f & ~s) == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        
        if best_mask is None:
            break
            
        family.append(best_mask)
        used.add(best_mask)
        
    # 3. Local Search (Hill Climbing) to escape local optima
    # Try swapping elements to improve score.
    # Since greedy is strong, we just do a few random swaps or targeted swaps.
    
    best_family = family[:]
    best_score = count_comparables(best_family)
    
    deadline = time.time() + 50
    
    # Targeted local search: 
    # For each element in family, try replacing it with the best candidate not in family.
    # This is O(m * N * m) per pass.
    # m=40, N=4096. 40 * 4096 * 40 = 6.5M.
    # We can do ~10 passes in 50s.
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(best_family)):
            if time.time() > deadline:
                break
                
            old_mask = best_family[i]
            # Temporarily remove old_mask
            # Find best replacement
            best_gain = -1
            best_repl = None
            
            # Calculate gain for each candidate s not in family
            # Gain(s) = comparables(s, family \ {old_mask}) - comparables(old_mask, family \ {old_mask})
            # Note: comparables(old_mask, family \ {old_mask}) is fixed for this step.
            
            old_comp = 0
            for j in range(len(best_family)):
                if j == i:
                    continue
                if (old_mask & ~best_family[j]) == 0 or (best_family[j] & ~old_mask) == 0:
                    old_comp += 1
            
            for s in all_masks:
                if s in used: # 'used' contains old_mask, so we need to be careful.
                    # Actually, s should not be in family.
                    # If s is in family and s != old_mask, skip.
                    # If s == old_mask, skip (already removed from consideration effectively).
                    # Let's just check if s is in the set of current family masks.
                    pass
                
            # Better: maintain a set of family masks for O(1) lookup.
            fam_set = set(best_family)
            
            for s in all_masks:
                if s in fam_set:
                    continue
                
                new_comp = 0
                for j in range(len(best_family)):
                    if j == i:
                        continue
                    if (s & ~best_family[j]) == 0 or (best_family[j] & ~s) == 0:
                        new_comp += 1
                
                delta = new_comp - old_comp
                if delta > best_gain:
                    best_gain = delta
                    best_repl = s
            
            if best_repl is not None and best_gain > 0:
                # Swap
                best_family[i] = best_repl
                # Update used set? 
                # We don't strictly need 'used' if we use fam_set in the loop, 
                # but 'used' was for the greedy part.
                # For local search, we just check against fam_set.
                improved = True
        
        if not improved:
            break
            
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result

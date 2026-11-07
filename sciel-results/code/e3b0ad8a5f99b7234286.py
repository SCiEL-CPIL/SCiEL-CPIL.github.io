import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute nothing heavy, just use bitwise ops.
    # However, to speed up the "Gain" calculation, we can precompute
    # the list of all masks that are comparable to a given mask?
    # No, N=4096 is small enough to just loop.
    
    # Helper: Count comparable pairs in a family of masks
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if (family[i] & family[j]) == family[i] or (family[i] & family[j]) == family[j]:
                    count += 1
        return count

    # Precompute for each mask M, a 40-bit integer where bit k is set
    # if M is comparable to family[k]. This lets us compute the gain/loss
    # as a single bit_count instead of looping over all family members.
    def build_comparability_masks(family, m, N, n):
        """For each mask in [0, N), build a bitmask of which family indices it's comparable to."""
        comp_masks = [0] * N
        for M in range(N):
            bm = 0
            for k in range(m):
                f = family[k]
                if (M & f) == M or (M & f) == f:
                    bm |= (1 << k)
            comp_masks[M] = bm
        return comp_masks

    # Greedy Initialization
    # Start with a maximal chain to ensure high initial connectivity
    # Chain: 0, 1, 3, 7, ..., 4095
    chain = []
    cur = 0
    for b in range(n):
        chain.append(cur)
        cur |= (1 << b)
    chain.append(cur) # This adds the full set at the end if we started with 0 and added bits
    # Wait, the loop above:
    # b=0: appends 0, cur becomes 1
    # b=1: appends 1, cur becomes 3
    # ...
    # b=11: appends 2047, cur becomes 4095
    # Then appends 4095.
    # So chain has 13 elements: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095.
    # All are comparable.
    
    family = chain[:]
    fam_set = set(family)
    
    # Greedily add remaining m - len(family) sets
    while len(family) < m:
        if time.time() > deadline - 5:
            break
            
        best_mask = -1
        best_score = -1
        
        # Find mask M not in family maximizing comparability with current family
        for M in range(N):
            if M in fam_set:
                continue
            
            score = 0
            for f in family:
                if (f & M) == f or (f & M) == M:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_mask = M
        
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        
    # Local Search (Hill Climbing) with precomputed comparability masks
    # For speed: precompute for each candidate mask M, a bitmask of which
    # family indices it is comparable to. Then gain/loss = bit_count minus 1 (for self).
    
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2:
        improved = False
        passes += 1
        
        # Rebuild comparability masks at start of each pass (family may have changed)
        comp_masks = build_comparability_masks(family, m, N, n)
        
        for i in range(m):
            if time.time() > deadline - 2:
                break
                
            old_mask = family[i]
            
            # Loss: how many other family members is old_mask comparable to?
            # comp_masks[old_mask] has bit k set if old_mask comparable to family[k].
            # Bit i is set (self), so loss = bit_count - 1
            loss = comp_masks[old_mask].bit_count() - 1
            
            best_mask = -1
            best_gain = -1
            
            # Try all candidates M not in family
            for M in range(N):
                if M in fam_set:
                    continue
                
                # Gain: how many family members is M comparable to?
                # Bit i will be 0 since M != old_mask and M not in fam_set,
                # but we need to exclude position i from the count.
                # Actually, M is not in family, so bit i won't be set for M vs family[i]=old_mask
                # unless M is comparable to old_mask. But we're replacing old_mask with M,
                # so we want comparability of M with family[j] for j != i.
                gain = comp_masks[M].bit_count()
                # If M happens to be comparable to old_mask (family[i]), bit i is set,
                # but we're removing old_mask, so we should subtract 1 if bit i is set.
                if (comp_masks[M] >> i) & 1:
                    gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_mask = M
            
            # If we found a better replacement
            if best_mask != -1 and best_gain > loss:
                # Update family and fam_set
                fam_set.remove(old_mask)
                fam_set.add(best_mask)
                family[i] = best_mask
                improved = True
                
    # Final polish: run additional full passes with the fast bit_count method
    # until no improvement or deadline reached
    while time.time() < deadline - 1:
        improved = False
        comp_masks = build_comparability_masks(family, m, N, n)
        
        for i in range(m):
            if time.time() > deadline - 1:
                break
                
            old_mask = family[i]
            loss = comp_masks[old_mask].bit_count() - 1
            
            best_mask = -1
            best_gain = -1
            
            for M in range(N):
                if M in fam_set:
                    continue
                gain = comp_masks[M].bit_count()
                if (comp_masks[M] >> i) & 1:
                    gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = M
            
            if best_mask != -1 and best_gain > loss:
                fam_set.remove(old_mask)
                fam_set.add(best_mask)
                family[i] = best_mask
                improved = True
        
        if not improved:
            break
    
    # Convert to lists of integers
    result = []
    for mask in family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result

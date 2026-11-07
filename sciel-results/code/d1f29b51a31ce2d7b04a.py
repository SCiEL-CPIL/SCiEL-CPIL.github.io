import time
import sys

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    start_time = time.time()
    deadline = start_time + 75  # Safety margin for 90s kill
    
    # 1. Initialize with the longest chain (length 13)
    family = []
    cur = 0
    family.append(cur)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    used = set(family)
    
    # 2. Greedily add sets that maximize comparable pairs with current family
    for _ in range(m - len(family)):
        best_gain = -1
        best_mask = None
        
        for s in range(total_masks):
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
    
    # 3. Local Search
    best_family = family[:]
    
    def count_comparables(fam):
        cnt = 0
        L = len(fam)
        for i in range(L):
            for j in range(i + 1, L):
                if (fam[i] & ~fam[j]) == 0 or (fam[j] & ~fam[i]) == 0:
                    cnt += 1
        return cnt

    current_score = count_comparables(best_family)
    
    # Local search with swap moves
    while time.time() < deadline:
        improved = False
        
        # Compute contribs for current family
        contribs = [0] * m
        for i in range(m):
            fi = best_family[i]
            for j in range(m):
                if i != j:
                    fj = best_family[j]
                    if (fi & ~fj) == 0 or (fj & ~fi) == 0:
                        contribs[i] += 1
        
        # Precompute for each candidate s the sum of comparabilities with current family
        fam_set = set(best_family)
        sum_comp = [0] * total_masks
        for s in range(total_masks):
            if s in fam_set:
                continue
            sc = 0
            for f in best_family:
                if (s & ~f) == 0 or (f & ~s) == 0:
                    sc += 1
            sum_comp[s] = sc
            
        # Now find best swap
        best_delta = 0
        best_i = -1
        best_s = -1
        
        for i in range(m):
            fi = best_family[i]
            for s in range(total_masks):
                if s in fam_set:
                    continue
                comp_sf = 1 if ((s & ~fi) == 0 or (fi & ~s) == 0) else 0
                val = sum_comp[s] - comp_sf
                delta = val - contribs[i]
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_s = s
        
        if best_delta > 0 and best_i != -1 and best_s != -1:
            # Perform swap
            best_family[best_i] = best_s
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

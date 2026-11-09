import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    mask_to_combo = {}
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        mask_to_combo[mask] = combo
        
    # Precompute complement masks for quick R calculation?
    # R = all_mask ^ (A | C)
    all_mask = (1 << n) - 1

    def can_add(C, family, family_set):
        """
        Check if C can be added to family without forming a sunflower
        with any pair in family.
        """
        for A in family:
            K = A & C
            # R is the set of elements not in A and not in C
            R = all_mask ^ (A | C)
            
            # We need to check if there exists B in family such that:
            # 1. B contains K
            # 2. B \ K is a subset of R
            # 3. |B| = 4
            # This implies B is formed by K and (4 - |K|) elements from R.
            
            k_size = bin(K).count('1')
            needed = 4 - k_size
            
            if needed == 0:
                # B must be exactly K. But |K|=4 implies A=C=K, 
                # but A and C are distinct in the loop? 
                # No, A is from family, C is new. 
                # If A=C, they are not distinct sets in the sunflower definition?
                # "No three distinct sets". A and C are distinct.
                # If A=C, K=A. R = all ^ A.
                # B must contain A. |B|=4. So B=A.
                # But B must be in family. A is in family.
                # Does {A, A, C} form a sunflower? No, sets must be distinct.
                # So if B=A, the triple is {A, C, A} which has duplicates.
                # So we should check B != A and B != C.
                # If needed == 0, B=K=A. B is A. 
                # The triple is A, C, A. Not distinct. So no conflict.
                continue

            if needed == 1:
                # B = K | (1 << x) for x in R
                # Iterate bits in R
                r = R
                while r:
                    lsb = r & -r
                    b_mask = K | lsb
                    if b_mask in family_set:
                        # Check distinctness
                        if b_mask != A and b_mask != C:
                            return False
                    r ^= lsb
            elif needed == 2:
                # B = K | (1 << x) | (1 << y) for x, y in R, x < y
                # Get list of bits in R
                bits = []
                r = R
                while r:
                    lsb = r & -r
                    bits.append(lsb)
                    r ^= lsb
                for i in range(len(bits)):
                    for j in range(i + 1, len(bits)):
                        b_mask = K | bits[i] | bits[j]
                        if b_mask in family_set:
                            if b_mask != A and b_mask != C:
                                return False
            elif needed == 3:
                bits = []
                r = R
                while r:
                    lsb = r & -r
                    bits.append(lsb)
                    r ^= lsb
                for i in range(len(bits)):
                    for j in range(i + 1, len(bits)):
                        for l in range(j + 1, len(bits)):
                            b_mask = K | bits[i] | bits[j] | bits[l]
                            if b_mask in family_set:
                                if b_mask != A and b_mask != C:
                                    return False
            elif needed == 4:
                bits = []
                r = R
                while r:
                    lsb = r & -r
                    bits.append(lsb)
                    r ^= lsb
                for i in range(len(bits)):
                    for j in range(i + 1, len(bits)):
                        for l in range(j + 1, len(bits)):
                            for m in range(l + 1, len(bits)):
                                b_mask = K | bits[i] | bits[j] | bits[l] | bits[m]
                                if b_mask in family_set:
                                    if b_mask != A and b_mask != C:
                                        return False
            
        return True

    best_family_size = 0
    best_family = []
    
    # Try multiple random restarts
    max_restarts = 5000 # Can increase due to speedup
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_masks)
        
        family = []
        family_set = set()
        
        for C in all_masks:
            if time.time() > deadline:
                break
                
            if can_add(C, family, family_set):
                family.append(C)
                family_set.add(C)
                
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
            if best_family_size >= 250:
                break

    if not best_family:
        best_family = [all_masks[0]]

    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result

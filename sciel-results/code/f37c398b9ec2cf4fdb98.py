import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    # Generate all k-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    best_family = []
    start_time = time.time()
    max_time = 50.0 # Leave buffer for overhead
    
    # Optimization: Precompute complements? No, n=16 is small.
    # Use bitwise operations for fast conflict checking.
    
    while time.time() - start_time < max_time:
        random.shuffle(all_sets)
        family = []
        family_masks = [] # List of masks in family
        
        for s in all_sets:
            valid = True
            # Check if s can be added to family
            # Condition: No A, B in family such that A cap B = A cap S = B cap S
            # For each A in family, let I = A cap S.
            # We need to ensure no B in family satisfies:
            # 1. B contains I
            # 2. B is disjoint from (A xor S)
            
            for a in family_masks:
                I = a & s
                E = a ^ s # (A \ I) U (S \ I)
                
                # Check if any B in family is a superset of I and disjoint from E
                # We can iterate through family_masks
                for b in family_masks:
                    if (b & I) == I and (b & E) == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(s)
                family_masks.append(s)
        
        if len(family) > len(best_family):
            best_family = family
            # If we found a very large family, we might want to try more restarts or stop early?
            # The max possible is likely around 200-250.
            # Let's keep running to see if we can improve.
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result

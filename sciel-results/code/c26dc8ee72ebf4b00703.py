import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    best_family = []
    
    # We can try multiple random orders
    # Each greedy pass is fast, so we can do many.
    
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        
        for S in all_sets:
            if time.time() > deadline:
                break
                
            # Check if S can be added
            valid = True
            groups = {}
            
            # Build groups of existing sets based on intersection with S
            for A in family:
                Z = A & S
                if Z not in groups:
                    groups[Z] = []
                groups[Z].append(A)
                
            # Check for sunflowers
            for Z, L in groups.items():
                if len(L) >= 2:
                    # Check pairs in L
                    # Optimization: if len(L) is large, it's expensive.
                    # But L size is usually small.
                    for i in range(len(L)):
                        for j in range(i + 1, len(L)):
                            if (L[i] & L[j]) == Z:
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
                    
            if valid:
                family.append(S)
                
        if len(family) > len(best_family):
            best_family = family
            
    # Convert bitmasks to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

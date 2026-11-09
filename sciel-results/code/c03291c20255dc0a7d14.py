# EVOLVE-BLOCK-START
from itertools import combinations
import random
import time

def construct():
    # Generate all 4-subsets as integers (bitmasks)
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    best_size = 0
    best_family = []
    
    rng = random.Random(42)
    
    start_time = time.time()
    max_time = 55.0
    
    # We will run multiple greedy trials
    # The core improvement is the optimized conflict check using grouping by intersection
    
    while time.time() - start_time < max_time:
        order = list(range(n_sets))
        rng.shuffle(order)
        
        current_family = []
        
        for idx in order:
            s = all_sets[idx]
            conflict_found = False
            
            # Group existing sets in current_family by their intersection with s
            # Since s has 4 bits, there are at most 16 possible intersection values
            groups = {}
            
            for x in current_family:
                inter = x & s
                if inter in groups:
                    groups[inter].append(x)
                else:
                    groups[inter] = [x]
            
            # Check for sunflower conflicts within each group
            for c, group_list in groups.items():
                if len(group_list) < 2:
                    continue
                
                # Check pairs within this group
                # If x_i & x_j == c, then (x_i, x_j, s) form a sunflower with core c
                l_len = len(group_list)
                for i in range(l_len):
                    xi = group_list[i]
                    for j in range(i + 1, l_len):
                        xj = group_list[j]
                        if (xi & xj) == c:
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break
            
            if not conflict_found:
                current_family.append(s)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
    # Convert best_family masks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(16):
            if (mask >> i) & 1:
                subset.append(i + 1)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END

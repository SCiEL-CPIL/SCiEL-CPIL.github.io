import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 0..15 corresponding to 1..16
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    best_family = []
    best_size = 0
    
    # Use multiple random orderings
    # We can also try a fixed "smart" order first, e.g., sorted by mask value
    # But random is better for escaping local optima
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(all_sets)
        family = []
        
        for mask in all_sets:
            if time.time() > deadline:
                break
            
            # Check if mask can be added to family
            # Conflict if exists a, b in family such that a&b == a&mask == b&mask
            compatible = True
            len_f = len(family)
            # Optimization: If family is small, check is fast.
            # We iterate pairs (i, j) with i > j
            for i in range(len_f):
                a = family[i]
                am = a & mask
                for j in range(i):
                    b = family[j]
                    bm = b & mask
                    ab = a & b
                    if am == bm == ab:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                family.append(mask)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
            # Early exit if we hit the cap
            if best_size >= 250:
                break
        else:
            # If we are not improving, we can continue shuffling
            # The random shuffle changes the greedy path
            pass

    # Convert best_family back to list of sorted lists of integers 1..16
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result

import itertools
import time

def construct():
    start_time = time.time()
    LIMIT = 55.0
    
    p = 101
    denominators = list(range(1, 11))
    n = 10
    
    # Precompute modular inverses
    inverses = {x: pow(x, -1, p) for x in denominators}
    
    # Precompute sum of inverses for every non-empty subset
    # Store as: residue -> list of (size, combo) sorted by size
    # Actually, we want for each residue, the minimum size subset.
    
    # Process by size: for each size from 1 to n, check all combos of that size
    # For each residue, record the first (smallest size) combo that hits it.
    
    best_size = [n + 1] * p
    best_combo = [None] * p
    
    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            if best_size[residue] > size:
                best_size[residue] = size
                best_combo[residue] = combo
        
        # Check if all residues are covered
        if all(sz <= size for sz in best_size):
            break
        
        # Time guard
        if time.time() - start_time > LIMIT:
            break
    
    # Construct result
    result = []
    for r in range(p):
        if best_combo[r] is not None:
            result.append(list(best_combo[r]))
        else:
            # Fallback: find any subset that sums to r
            # This shouldn't happen, but use full set as last resort
            result.append(denominators[:])
    
    return result

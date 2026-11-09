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
    
    # We want to find, for each residue r in 0..100,
    # a subset of denominators with minimum size such that sum(inv(d)) % p == r.
    # We check sizes 1, 2, 3, ... in increasing order.
    
    # assignments[r] = tuple of denominators
    assignments = {}
    
    # To ensure we find the MINIMUM size for each residue,
    # we must iterate by size.
    # We stop processing size k only if all residues are covered.
    # However, a residue might be covered by size 2, but we process size 1 first.
    # So we process size 1, assign residues not yet assigned.
    # Then size 2, assign residues not yet assigned.
    # This guarantees min size.
    
    # Optimization: Precompute sums for all subsets of size up to some limit?
    # Or just iterate.
    
    # Since n=10, total subsets 2^10 - 1 = 1023.
    # We can generate all, sort by length, then iterate.
    
    all_subsets = []
    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            all_subsets.append(combo)
            
    # all_subsets is already sorted by size (size 1, then 2, etc.)
    # within same size, lex order.
    
    for combo in all_subsets:
        s = 0
        for d in combo:
            s += inverses[d]
        residue = s % p
        
        if residue not in assignments:
            assignments[residue] = combo
            
        # If all residues are covered, we can stop looking for larger sizes
        if len(assignments) == p:
            break
            
        if (len(assignments) % 50 == 0) and (time.time() - start_time > LIMIT):
            break
            
    # Construct result
    result = []
    max_len = 0
    for r in range(p):
        if r in assignments:
            lst = list(assignments[r])
            result.append(lst)
            if len(lst) > max_len:
                max_len = len(lst)
        else:
            # Fallback (should not happen)
            lst = list(denominators)
            result.append(lst)
            if len(lst) > max_len:
                max_len = len(lst)
                
    return result

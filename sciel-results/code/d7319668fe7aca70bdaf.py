import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    p = 101
    denominators = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denominators]
    
    # Map residue to list of (length, subset_tuple)
    residue_to_subsets = {r: [] for r in range(p)}
    
    # Precompute all non-empty subsets
    # 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                terms.append(denominators[i])
                s += invs[i]
        s %= p
        length = len(terms)
        residue_to_subsets[s].append((length, tuple(terms)))
        
    # For each residue, find the minimum length subset
    # This minimizes the maximum row length
    assignment = {}
    for r in range(p):
        if residue_to_subsets[r]:
            # Find min length
            min_len = min(item[0] for item in residue_to_subsets[r])
            # Pick the first one with min length (deterministic tie-break by tuple)
            best_subset = None
            for item in residue_to_subsets[r]:
                if item[0] == min_len:
                    if best_subset is None or item[1] < best_subset:
                        best_subset = item[1]
            assignment[r] = list(best_subset)
        else:
            # Fallback (should not happen)
            assignment[r] = [1]
            
    return [assignment[r] for r in range(p)]

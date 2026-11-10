import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # Precompute all subsets and their residues, grouped by size
    # For each size s, store a dict: residue -> subset (first one found)
    example_by_size = [dict() for _ in range(11)]
    
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            res = sum(inv[x] for x in combo) % 101
            if res not in example_by_size[s]:
                example_by_size[s][res] = list(combo)
    
    # For each residue, find the smallest size s that achieves it
    # Since rows are independent (denominators can recur across rows),
    # picking the minimum size per residue is globally optimal.
    
    result = []
    for r in range(101):
        found = None
        for s in range(1, 11):
            if r in example_by_size[s]:
                found = example_by_size[s][r]
                break
        if found is None:
            # Fallback: use all 10 denominators
            found = list(range(1, 11))
        result.append(found)
    
    return result

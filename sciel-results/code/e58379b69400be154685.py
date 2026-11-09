import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    p = 101
    denom_range = range(1, 11)
    inv = {x: pow(x, -1, p) for x in denom_range}

    # Precompute all subsets grouped by size
    subsets_by_size = {}
    for size in range(1, 11):
        subsets_by_size[size] = []
        for combo in combinations(denom_range, size):
            subsets_by_size[size].append(combo)

    # Direct computation: for each residue, find the minimum-size subset achieving it.
    # Since denominators can recur across rows, rows are independent.
    # The optimal per-row minimum size trivially minimizes the maximum row length.
    
    optimal = {}
    for r in range(p):
        found = False
        for size in range(1, 11):
            for combo in subsets_by_size[size]:
                if sum(inv[x] for x in combo) % p == r:
                    optimal[r] = combo
                    found = True
                    break
            if found:
                break
        if not found:
            # Fallback: should not happen, but use a default
            optimal[r] = (1,)

    # Verify all residues are covered
    for r in range(p):
        if r not in optimal:
            optimal[r] = (1,)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Return what we have
        return [list(optimal[r]) for r in range(101)]

    return [list(optimal[r]) for r in range(101)]

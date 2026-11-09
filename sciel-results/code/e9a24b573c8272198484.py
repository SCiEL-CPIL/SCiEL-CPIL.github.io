"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    p = 101
    n = 10
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, -1, p)
    
    # For each residue, find the minimum-size subset achieving it
    # Since rows are independent (denominators can recur across rows),
    # the optimal max row length = max over residues of min subset size
    # We enumerate by size to find minimum-size subset per residue
    
    # Precompute all subset sums grouped by size
    # For each size k from 1 to 10, iterate all C(10,k) subsets
    # and record which residues are achievable
    
    best_subset = {}  # residue -> (size, subset_list)
    
    # Enumerate subsets by increasing size
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        # Generate all combinations of size k from 1..10
        from itertools import combinations
        for combo in combinations(range(1, n + 1), k):
            s = sum(inv[x] for x in combo) % p
            if s not in best_subset:
                best_subset[s] = (k, list(combo))
        
        # If we've covered all residues, we can stop
        if len(best_subset) == p:
            break
    
    # Build the result: 101 nonempty lists
    result = []
    for r in range(p):
        if r in best_subset:
            result.append(best_subset[r][1])
        else:
            # Fallback: should not happen, but use a valid nonempty subset
            result.append([1])
    
    return result
# EVOLVE-BLOCK-END

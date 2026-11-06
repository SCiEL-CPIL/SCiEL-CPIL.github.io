"""Search-based program: find minimum max row length covering all residues mod 101."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    P = 101
    D = list(range(1, 11))
    
    # Explicit deadline check (not expected to trigger)
    if time.time() - t0 > deadline:
        return [list(range(1, 11))] * 101
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # Build all nonempty subsets with their residues, grouped by size
    subsets_by_size = {i: [] for i in range(1, 11)}
    
    for size in range(1, 11):
        for combo in combinations(D, size):
            res = sum(inv[x] for x in combo) % P
            subsets_by_size[size].append((res, list(combo)))
    
    # Find minimum k such that subsets of size <= k cover all 101 residues
    # For each k, check if the union of residues from sizes 1..k covers all residues
    best_k = None
    for k in range(1, 11):
        covered = set()
        for s in range(1, k + 1):
            for res, _ in subsets_by_size[s]:
                covered.add(res)
        if len(covered) == 101:
            best_k = k
            break
    
    if best_k is None:
        # Should not happen, but fallback
        best_k = 10
    
    # Now assign one subset to each residue, using subsets of size <= best_k
    # Prefer smaller subsets to leave flexibility
    assignment = [None] * 101
    
    # Process residues, for each residue pick a subset of size <= best_k
    # To minimize max size, we already know best_k is the minimum
    # Just assign any valid subset
    # Build a map: residue -> list of (size, terms) for sizes <= best_k
    residue_to_subsets = {r: [] for r in range(101)}
    for s in range(1, best_k + 1):
        for res, terms in subsets_by_size[s]:
            residue_to_subsets[res].append((s, terms))
    
    # Assign: for each residue, pick the smallest subset available
    for r in range(101):
        if residue_to_subsets[r]:
            # Pick the one with smallest size (first in list since we added by size)
            _, terms = residue_to_subsets[r][0]
            assignment[r] = terms
        else:
            # Should not happen if best_k is correct
            assignment[r] = [1]
    
    # Verify
    for r in range(101):
        s = sum(inv[x] for x in assignment[r]) % P
        if s != r:
            # Fallback: try to find correct one
            for sz, terms in residue_to_subsets[r]:
                assignment[r] = terms
                break
    
    return assignment

# EVOLVE-BLOCK-END

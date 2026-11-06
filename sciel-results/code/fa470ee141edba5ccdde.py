"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # Deadline guard: we have 90s wall-clock; use 80s as safety margin.
    deadline = time.time() + 80.0
    
    p = 101
    denominators = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in denominators:
        inv[x] = pow(x, -1, p)
        
    # Enumerate all non-empty subsets of {1..10}, grouped by size.
    # For each size k, we store a list of (residue, subset_list).
    subsets_by_size = {k: [] for k in range(1, 11)}
    
    for size in range(1, 11):
        for comb in combinations(denominators, size):
            s = sum(inv[x] for x in comb) % p
            subsets_by_size[size].append((s, list(comb)))
            
    # Determine the minimum K such that all residues are covered by subsets of size <= K.
    covered = [False] * p
    min_k_required = 10
    
    for k in range(1, 11):
        for res, subset in subsets_by_size[k]:
            covered[res] = True
        
        if all(covered):
            min_k_required = k
            break
            
    # Now we need to assign to each residue r a subset of size <= min_k_required
    # that sums to r mod p. We want to minimize the maximum row length.
    # Since min_k_required is the theoretical minimum for the max length,
    # any valid assignment with all rows having length <= min_k_required is optimal.
    
    # However, we should try to minimize the maximum length further if possible.
    # Actually, by definition of min_k_required, we cannot have all rows with length < min_k_required.
    # So the optimal max length is exactly min_k_required.
    
    # Strategy: For each residue, find the smallest subset (by size) that sums to it.
    # This ensures we use the shortest possible subsets, which may help in case
    # some residues only have representations of size exactly min_k_required.
    
    result = [None] * p
    found_for_residue = [False] * p
    
    # Iterate by increasing size; first hit for a residue is its smallest representation.
    for k in range(1, min_k_required + 1):
        for res, subset in subsets_by_size[k]:
            if not found_for_residue[res]:
                result[res] = subset
                found_for_residue[res] = True
                
        # Early exit if all found
        if all(found_for_residue):
            break
            
    # Safety: if any residue wasn't found (shouldn't happen), use any subset
    for r in range(p):
        if not found_for_residue[r]:
            for k in range(1, 11):
                for res, subset in subsets_by_size[k]:
                    if res == r:
                        result[r] = subset
                        found_for_residue[r] = True
                        break
                if found_for_residue[r]:
                    break
            if not found_for_residue[r]:
                raise RuntimeError(f"Residue {r} not covered")
                
    return result

# EVOLVE-BLOCK-END

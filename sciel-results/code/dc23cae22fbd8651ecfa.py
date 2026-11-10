"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill
    
    MOD = 101
    DENOMS = list(range(1, 11))
    N = len(DENOMS)
    
    # Precompute inverses
    invs = [pow(x, -1, MOD) for x in DENOMS]
    
    # min_len_for_residue[r] will store the minimum length of a subset summing to r
    # best_subset_for_residue[r] will store the actual subset (as a list of denominators)
    min_len_for_residue = [11] * MOD
    best_subset_for_residue = [None] * MOD
    
    # Iterate over all non-empty subsets of denominators, grouped by size
    # This allows early termination once all residues are covered at the current size
    for size in range(1, N + 1):
        if time.time() > deadline:
            break
        for combo in combinations(range(N), size):
            s = 0
            for i in combo:
                s += invs[i]
            s %= MOD
            
            if size < min_len_for_residue[s]:
                min_len_for_residue[s] = size
                best_subset_for_residue[s] = [DENOMS[i] for i in combo]
        
        # Check if all residues are now covered
        if all(min_len_for_residue[r] <= size for r in range(MOD)):
            break
    
    # Build the result
    result = []
    for r in range(MOD):
        if best_subset_for_residue[r] is not None:
            result.append(list(best_subset_for_residue[r]))
        else:
            # Fallback: This should not happen if the problem is solvable.
            # If it does, we return an empty list, which will likely fail validation.
            result.append([])
            
    return result
# EVOLVE-BLOCK-END

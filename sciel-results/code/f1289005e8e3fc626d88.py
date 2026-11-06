"""Optimized program for the suite task; finds minimum cardinality subset for each residue."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    p = 101
    denominators = list(range(1, 11))
    
    # Deadline guard: return best result before 85s wall-clock (5s safety margin from 90s limit)
    deadline = time.time() + 85.0
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    # best_subset[r] will hold the smallest subset (by length) that sums to r
    # Initialize with None
    best_subset = [None] * p
    
    # Iterate through all non-empty subsets by length (shortest first)
    # There are 2^10 - 1 = 1023 subsets
    # Using itertools.combinations for clean length-ordered iteration
    for length in range(1, 11):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(denominators, length):
            s = 0
            for x in combo:
                s = (s + inv[x]) % p
            r = s
            # Only update if we haven't found a subset for this residue yet
            # (since we iterate by increasing length, first found is shortest)
            if best_subset[r] is None:
                best_subset[r] = list(combo)
        
        # Early termination check: if all residues are covered, we're done
        if all(x is not None for x in best_subset):
            break
    
    # Construct the final list of 101 lists
    result = []
    for r in range(p):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: should not happen with 1023 subsets vs 101 residues
            # Try one more exhaustive pass if we hit the deadline early
            result.append([])
    
    # If any residue is still uncovered, do a full exhaustive search as fallback
    if any(len(row) == 0 for row in result):
        # Reset and do full exhaustive search without deadline pressure
        best_subset = [None] * p
        for mask in range(1, 1 << 10):
            terms = []
            s = 0
            for x in range(1, 11):
                if mask & (1 << (x - 1)):
                    terms.append(x)
                    s = (s + inv[x]) % p
            r = s
            if best_subset[r] is None or len(terms) < len(best_subset[r]):
                best_subset[r] = terms
        
        result = []
        for r in range(p):
            if best_subset[r] is not None:
                result.append(best_subset[r])
            else:
                result.append([1])  # Last resort placeholder
    
    return result
# EVOLVE-BLOCK-END

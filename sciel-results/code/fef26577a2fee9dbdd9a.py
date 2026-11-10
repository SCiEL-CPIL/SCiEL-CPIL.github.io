import time

def construct():
    # The problem is small enough (2^10 subsets) that brute force is optimal and instantaneous.
    # We enumerate all non-empty subsets of {1, ..., 10}.
    # For each subset, we calculate the sum of inverses mod 101.
    # For each residue, we track the minimum size of a subset that produces it.
    # We return the list of subsets (as lists of denominators) for each residue.
    
    p = 101
    max_den = 10
    
    # Precompute inverses
    inv = [0] * (max_den + 1)
    for i in range(1, max_den + 1):
        inv[i] = pow(i, -1, p)
        
    # min_size[r] stores the minimum size of a subset summing to r
    # best_subset[r] stores the subset (as a tuple for hashability/storage)
    min_size = [float('inf')] * p
    best_subset = [None] * p
    
    # Iterate over all non-empty masks
    # mask bit i (0-indexed) corresponds to denominator i+1
    for mask in range(1, 1 << max_den):
        s = 0
        size = 0
        terms = []
        for i in range(1, max_den + 1):
            if mask & (1 << (i - 1)):
                s = (s + inv[i]) % p
                size += 1
                terms.append(i)
                
        # Update if this subset is smaller for this residue
        if size < min_size[s]:
            min_size[s] = size
            best_subset[s] = tuple(terms)
            
    # Construct result
    # If a residue is not covered (should not happen), we need a fallback.
    # But with 1023 subsets and 101 residues, coverage is dense.
    result = []
    for r in range(p):
        if best_subset[r] is not None:
            result.append(list(best_subset[r]))
        else:
            # Fallback: This should never happen. 
            # If it does, we can't satisfy "non-empty". 
            # But we must return something. 
            # A single denominator 1 sums to 1. 
            # If r is not covered, the problem instance is unsolvable as stated, 
            # but we assume it is solvable.
            result.append([1]) # Dummy, likely invalid if r != 1, but prevents crash
            
    return result

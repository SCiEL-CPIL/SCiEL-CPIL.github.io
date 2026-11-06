import time
from itertools import combinations

def construct():
    deadline = time.time() + 80
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # Precompute sums for all subsets
    # subset_sums[residue] = min_size
    min_size = [11] * 101
    min_mask = [0] * 101
    
    # Generate subsets by size
    for k in range(1, 11):
        for comb in combinations(range(1, 11), k):
            s = sum(inv[x] for x in comb) % 101
            if min_size[s] > k:
                min_size[s] = k
                # Store the combination or mask for reconstruction
                mask = 0
                for x in comb:
                    mask |= (1 << (x-1))
                min_mask[s] = mask
                
        # Early termination check: if all are covered, we can stop?
        # No, we want the MIN size. We iterate k=1, then 2, etc.
        # If we find a cover at k=4, we don't need to check k=5 for those residues.
        # But we need to check if ALL are covered.
        # If after k=4, some are still 11, we must go to k=5.
        
    # Determine max row length
    max_len = max(min_size)
    
    # If we want to be fancy, we can binary search K.
    # But since we already have min_size for each, max_len is the answer.
    
    result = []
    for r in range(101):
        mask = min_mask[r]
        terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
        result.append(terms)
        
    return result

import time

def construct():
    t0 = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    P = 101
    denominators = list(range(1, 11))
    
    # Precompute inverses
    inv = [0] * 11
    for x in denominators:
        inv[x] = pow(x, -1, P)
        
    # Generate all non-empty subsets
    # subset_info: list of (size, residue, subset_list)
    subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        size = 0
        for x in denominators:
            if mask & (1 << (x - 1)):
                terms.append(x)
                res = (res + inv[x]) % P
                size += 1
        subsets.append((size, res, terms))
        
    # Sort subsets by size to prioritize smaller ones
    subsets.sort(key=lambda x: x[0])
    
    # Try k = 1, 2, ...
    # We need to find the smallest k such that all residues are covered by subsets of size <= k
    
    # To be efficient, we can just iterate through subsets in increasing order of size
    # and assign them to residues if not yet assigned.
    # Once all 101 residues are assigned, the size of the last assigned subset is the max row length.
    # But we must ensure that the *minimum* possible max row length is achieved.
    # Since we process in increasing size, the first time we cover all residues, 
    # the current subset size is the optimal max row length.
    
    assignment = [None] * P
    covered_count = 0
    max_size_used = 0
    
    for size, res, terms in subsets:
        if assignment[res] is None:
            assignment[res] = terms
            covered_count += 1
            max_size_used = size  # Since sorted, this tracks the current max size
            if covered_count == P:
                # We have covered all residues using subsets of size <= max_size_used
                # This is the optimal solution because we processed in increasing size order.
                break
        
    if covered_count < P:
        # Fallback: should not happen as k=10 covers everything
        # Force fill with remaining subsets
        for size, res, terms in subsets:
            if assignment[res] is None:
                assignment[res] = terms
                covered_count += 1
                if covered_count == P:
                    break
                    
    # Check time
    if time.time() - t0 > deadline:
        # Return whatever we have, though it should be done instantly
        pass
        
    return assignment

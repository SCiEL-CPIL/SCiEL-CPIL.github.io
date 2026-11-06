import time

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 80.0
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # min_size_for_residue[r] stores the minimum size of a subset summing to r
    # subset_for_residue[r] stores that subset
    min_size_for_residue = [101] * 101
    subset_for_residue = [None] * 101
    
    # Iterate over all non-empty subsets of D
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        terms = []
        res = 0
        for x in D:
            if mask & (1 << (x - 1)):
                terms.append(x)
                res = (res + inv[x]) % P
        
        size = len(terms)
        if size < min_size_for_residue[res]:
            min_size_for_residue[res] = size
            subset_for_residue[res] = terms
    
    # Find the minimum K such that all residues are covered by subsets of size <= K
    for K in range(1, 11):
        covered = True
        for r in range(101):
            if min_size_for_residue[r] > K:
                covered = False
                break
        
        if covered:
            result = []
            for r in range(101):
                result.append(subset_for_residue[r])
            return result
    
    # Fallback (should not happen)
    return [subset_for_residue[r] for r in range(101)]

# EVOLVE-BLOCK-END

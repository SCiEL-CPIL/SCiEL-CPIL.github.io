"""Search-based program: find minimum max row length covering all residues mod 101."""
import time

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # Build all nonempty subsets with their residues and sizes
    # For each residue, track the minimum size subset that achieves it
    min_size_for_residue = [101] * 101  # 101 is "infinity" (larger than any possible size)
    subset_for_residue = [None] * 101
    
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
        # Check if all residues have a subset of size <= K
        covered = True
        for r in range(101):
            if min_size_for_residue[r] > K:
                covered = False
                break
        
        if covered:
            # Build the result: for each residue, use the subset that achieves it
            # We need to make sure we're using distinct subsets (but since each subset
            # has a unique residue, this is automatic)
            result = []
            for r in range(101):
                result.append(subset_for_residue[r])
            return result
    
    # Fallback: should not reach here, but use the seed approach
    representations = {}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        representations.setdefault(residue, terms)
    return [representations[r] for r in range(101)]

# EVOLVE-BLOCK-END

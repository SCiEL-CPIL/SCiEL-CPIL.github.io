import time

def construct():
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # Enumerate all non-empty subsets, find minimum-size subset for each residue
    min_size_for_residue = [11] * 101
    best_terms_for_residue = [None] * 101
    
    for mask in range(1, 1 << 10):
        terms = []
        residue = 0
        for i in range(10):
            if mask & (1 << i):
                d = i + 1
                terms.append(d)
                residue += inv[d]
        size = len(terms)
        residue %= 101
        if size < min_size_for_residue[residue]:
            min_size_for_residue[residue] = size
            best_terms_for_residue[residue] = terms
    
    # Verify all residues are covered
    for r in range(101):
        if min_size_for_residue[r] == 11:
            raise RuntimeError("Residue not covered")
    
    return [best_terms_for_residue[r] for r in range(101)]

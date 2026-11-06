"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inverses = [0] * 11
    for x in range(1, 11):
        inverses[x] = pow(x, -1, 101)
    
    # For each residue 0..100, find the minimum-size subset of {1..10}
    # whose inverse sum equals that residue mod 101.
    # We want to minimize the maximum row length across all 101 residues.
    
    # Enumerate all 2^10 - 1 = 1023 non-empty subsets
    # For each subset, compute its size and residue
    # For each residue, keep the smallest subset that achieves it
    
    best_size = [float('inf')] * 101
    best_terms = [None] * 101
    
    # Iterate by size to ensure we find minimum size first
    # Size 1: 10 subsets
    # Size 2: 45 subsets
    # Size 3: 120 subsets
    # etc.
    
    # We'll just iterate all masks and track min size per residue
    for mask in range(1, 1 << 10):
        # Compute size and residue
        size = 0
        res = 0
        terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                size += 1
                res = (res + inverses[x]) % 101
                terms.append(x)
        
        if size < best_size[res]:
            best_size[res] = size
            best_terms[res] = terms
    
    # Check that all residues are covered
    for r in range(101):
        if best_terms[r] is None:
            # This shouldn't happen, but handle gracefully
            best_terms[r] = [1]
    
    return [best_terms[r] for r in range(101)]
# EVOLVE-BLOCK-END

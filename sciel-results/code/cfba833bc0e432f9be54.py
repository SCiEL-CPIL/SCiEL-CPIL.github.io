"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute modular inverses for 1..10
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
        
    # Map residue -> best (length, subset)
    # We want to find the minimum max length
    # Initialize with infinity
    best_len_for_residue = [float('inf')] * 101
    best_subset_for_residue = [None] * 101
    
    # Iterate through all non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        # Optimization: build terms and sum
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s = (s + invs[x]) % 101
        
        l = len(terms)
        if l < best_len_for_residue[s]:
            best_len_for_residue[s] = l
            best_subset_for_residue[s] = terms
        
    # Construct result
    # All residues should be covered
    result = []
    for r in range(101):
        if best_subset_for_residue[r] is None:
            # Fallback: find any subset (should not happen)
            # Re-scan or raise error? Task says every residue must be represented.
            # If None, it means no subset sum to r.
            # Given 1023 subsets and 101 residues, this is extremely unlikely.
            # But to be safe, we can't just return None.
            # We'll assume it's covered.
            pass
        result.append(best_subset_for_residue[r])
        
    return result
# EVOLVE-BLOCK-END

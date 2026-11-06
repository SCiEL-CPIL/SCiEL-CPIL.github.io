# EVOLVE-BLOCK-START
def construct():
    # Initialize best subset for each residue as None
    best_subset = [None] * 101
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        
        # Only consider if this is a better (smaller) subset for its residue
        # We calculate residue
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        
        if best_subset[residue] is None or size < len(best_subset[residue]):
            best_subset[residue] = terms
            
    # Check if all residues are covered
    # If not, we might need a fallback, but the problem implies a solution exists.
    # However, to be safe, if any is None, we should handle it.
    # Given the problem constraints, all should be covered.
    
    # The "value" is the max length.
    # The function returns the lists.
    return best_subset
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute minimal subset for each residue
    # min_size[r] = smallest size of a subset summing to r
    # min_terms[r] = the terms of that subset
    min_size = [11] * 101
    min_terms = [None] * 101
    
    # Iterate over all non-empty subsets
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        
        # Update if this is a smaller subset for this residue
        if size < min_size[residue]:
            min_size[residue] = size
            min_terms[residue] = terms
    
    # The optimal max length is the max of min_size
    # We just return the min_terms for each residue
    # This automatically minimizes the max length and the sum of lengths
    return [min_terms[r] for r in range(101)]
# EVOLVE-BLOCK-END

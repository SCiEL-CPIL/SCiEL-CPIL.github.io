"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Denominators 1 to 10
    denoms = list(range(1, 11))
    n = len(denoms)
    # Precompute modular inverses
    invs = [pow(d, -1, 101) for d in denoms]
    
    # We want to find, for each residue r mod 101, a subset of denoms
    # whose inverses sum to r mod 101, minimizing the size of the subset.
    # 
    # Since there are only 2^10 = 1024 subsets, we can enumerate them all
    # and find the minimum-size subset for each residue.
    
    # For each residue, track the best (smallest size, then smallest mask) representation
    best_mask = [None] * 101  # best_mask[r] = mask achieving residue r with min size
    
    for mask in range(1, 1 << n):
        # Calculate sum of inverses for this mask
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s = (s + invs[idx]) % 101
            idx += 1
            m >>= 1
        
        size = bin(mask).count("1")
        
        if best_mask[s] is None:
            best_mask[s] = mask
        else:
            existing_size = bin(best_mask[s]).count("1")
            if size < existing_size or (size == existing_size and mask < best_mask[s]):
                best_mask[s] = mask
    
    # Build representations
    representations = [None] * 101
    for r in range(101):
        mask = best_mask[r]
        if mask is not None:
            terms = [d for i, d in enumerate(denoms) if (mask >> i) & 1]
            representations[r] = terms
        else:
            # Should not happen, but fallback
            representations[r] = [1]
    
    return representations
# EVOLVE-BLOCK-END

def construct():
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each residue, find the minimum-size subset
    min_size = [11] * 101  # 11 = infinity (max size is 10)
    best_subset = [None] * 101
    
    for mask in range(1, 1 << 10):
        # Count bits (size)
        size = bin(mask).count('1')
        # Compute residue
        residue = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                residue = (residue + inv[x - 1]) % 101
        
        if size < min_size[residue]:
            min_size[residue] = size
            best_subset[residue] = [x for x in range(1, 11) if mask & (1 << (x - 1))]
    
    # All residues should be covered
    assert all(s is not None for s in best_subset), "Not all residues covered"
    
    return best_subset

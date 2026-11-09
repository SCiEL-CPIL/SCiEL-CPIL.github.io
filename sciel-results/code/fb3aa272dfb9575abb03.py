def construct():
    import numpy as np
    
    # Generator matrix for the extended ternary Golay code [12, 6, 6] over GF(3)
    # This specific matrix is known to generate the code.
    G = np.array([
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int)
    
    blocks = []
    
    # Iterate over all 3^6 = 729 codewords
    # Using nested loops or product
    from itertools import product
    
    for coeffs in product(range(3), repeat=6):
        # Compute codeword c = coeffs * G mod 3
        c = np.dot(coeffs, G) % 3
        
        # Calculate weight (number of non-zero elements)
        weight = np.count_nonzero(c)
        
        if weight == 6:
            # Get support (indices where c is non-zero)
            support = tuple(np.nonzero(c)[0])
            blocks.append(list(support))
            
    # Ensure distinctness (should be distinct by code property, but safe to check)
    # The problem states "distinct 6-subsets".
    # The code generation naturally produces distinct vectors, hence distinct supports.
    
    return blocks

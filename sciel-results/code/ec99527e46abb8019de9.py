import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses Strassen's algorithm for each of the 5 independent 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    # W contributions for each product to C00, C01, C10, C11
    w_contrib = [
        [1, 0, 0, 0],   # P1 -> C00, C11
        [0, 0, 1, -1],  # P2 -> C10, C11
        [0, 1, 0, 1],   # P3 -> C01, C11
        [1, 0, 1, 0],   # P4 -> C00, C10
        [-1, 1, 0, 0],  # P5 -> C00, C01
        [0, 0, 0, 1],   # P6 -> C11
        [1, 0, 0, 0]    # P7 -> C00
    ]
    # Correction for P1: C00 +1, C11 +1
    w_contrib[0] = [1, 0, 0, 1]
    # Correction for P2: C10 +1, C11 -1
    w_contrib[1] = [0, 0, 1, -1]
    # Correction for P3: C01 +1, C11 +1
    w_contrib[2] = [0, 1, 0, 1]
    # Correction for P4: C00 +1, C10 +1
    w_contrib[3] = [1, 0, 1, 0]
    # Correction for P5: C00 -1, C01 +1
    w_contrib[4] = [-1, 1, 0, 0]
    # Correction for P6: C11 +1
    w_contrib[5] = [0, 0, 0, 1]
    # Correction for P7: C00 +1
    w_contrib[6] = [1, 0, 0, 0]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            for i in range(4):
                U[i][col] = strassen_A[r][i]
                
            # Fill V
            # B indices for block b:
            # e: j=0, k=2b -> idx 2b
            # f: j=0, k=2b+1 -> idx 2b+1
            # g: j=1, k=2b -> idx 10+2b
            # h: j=1, k=2b+1 -> idx 10+2b+1
            
            v_indices = [2*b, 2*b+1, 10+2*b, 10+2*b+1]
            for j_idx in range(4):
                V[v_indices[j_idx]][col] = strassen_B[r][j_idx]
                
            # Fill W
            # C indices for block b:
            # C00: k=2b, i=0 -> idx 4b
            # C01: k=2b+1, i=0 -> idx 4b+2
            # C10: k=2b, i=1 -> idx 4b+1
            # C11: k=2b+1, i=1 -> idx 4b+3
            
            w_indices = [4*b, 4*b+2, 4*b+1, 4*b+3]
            for c_idx in range(4):
                W[w_indices[c_idx]][col] = w_contrib[r][c_idx]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

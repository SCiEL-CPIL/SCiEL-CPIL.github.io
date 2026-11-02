"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen for 2x2x2 blocks
    # p=8, so 4 blocks of size 2
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A indices: 00, 01, 10, 11 -> U indices 0, 1, 2, 3
    A_coeffs = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # Strassen coefficients for B (2x2)
    # B local indices: e, f, g, h
    B_coeffs = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1]   # M7: g+h
    ]
    
    # Output combinations
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # W indices for block q (k=2q, 2q+1):
    # C_{0, 2q}   -> W[4q]
    # C_{1, 2q}   -> W[4q+1]
    # C_{0, 2q+1} -> W[4q+2]
    # C_{1, 2q+1} -> W[4q+3]
    
    W_combinations = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    for q in range(num_blocks):
        base_col = 7 * q
        # B indices in global V matrix
        # e = B[0, 2q] -> V[0*8 + 2q] = 2q
        # f = B[0, 2q+1] -> V[0*8 + 2q+1] = 2q+1
        # g = B[1, 2q] -> V[1*8 + 2q] = 8+2q
        # h = B[1, 2q+1] -> V[1*8 + 2q+1] = 8+2q+1
        b_indices = [2*q, 2*q+1, 8+2*q, 8+2*q+1]
        
        # W indices in global W matrix
        w_indices = [4*q, 4*q+1, 4*q+2, 4*q+3]
        
        for r in range(7):
            col = base_col + r
            
            # Fill U
            for i in range(4):
                U[i][col] = A_coeffs[r][i]
                
            # Fill V
            for i in range(4):
                V[b_indices[i]][col] = B_coeffs[r][i]
                
            # Fill W
            for i in range(4):
                W[w_indices[i]][col] = W_combinations[r][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

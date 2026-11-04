"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 multiplication (rank 49)
    # Part 2: 4x4x1 multiplication (rank 16)
    # Total rank 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0] * rank for _ in range(p * n)] # 20 x 65
    
    # --- Strassen 2x2x2 base decomposition ---
    # U_sub, V_sub, W_sub are lists of 7 vectors of length 4
    U_sub = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0],# m6
        [0, 1, 0, -1] # m7
    ]
    V_sub = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1],# m3
        [-1, 0, 1, 0],# m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    W_sub = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1],# m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0],# m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
    
    # Strassen 4x4x4 coefficients
    # k: 1..7
    # alpha: coeffs for A blocks (11, 12, 21, 22)
    # beta: coeffs for B blocks (11, 12, 21, 22)
    # gamma: coeffs for C blocks (11, 12, 21, 22)
    
    strassen_coeffs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # k=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # k=2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # k=3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # k=4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # k=5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # k=6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # k=7
    ]
    
    # Offsets for 2x2 blocks in 4x4 matrix (flattened)
    # Block 11: 0, 12: 4, 21: 8, 22: 12
    block_offsets = [0, 4, 8, 12]
    
    col = 0
    for k in range(7):
        alpha, beta, gamma = strassen_coeffs[k]
        for s in range(7):
            # Construct U_full, V_full, W_full
            U_full = [0] * 16
            V_full = [0] * 20 # 4x5
            W_full = [0] * 20 # 4x5
            
            # Fill U_full (16 elements)
            for b_idx in range(4):
                coeff = alpha[b_idx]
                if coeff != 0:
                    offset = block_offsets[b_idx]
                    for idx in range(4):
                        U_full[offset + idx] += coeff * U_sub[s][idx]
            
            # Fill V_full (20 elements)
            # B1 is 4x4 (indices 0-15), B2 is 4x1 (indices 16-19)
            # Strassen uses only B1 (the first 4 columns)
            for b_idx in range(4):
                coeff = beta[b_idx]
                if coeff != 0:
                    offset = block_offsets[b_idx]
                    for idx in range(4):
                        V_full[offset + idx] += coeff * V_sub[s][idx]
                        # Note: B2 part is 0 for this step
            
            # Fill W_full (20 elements)
            # C1 is 4x4 (indices 0-15), C2 is 4x1 (indices 16-19)
            # Strassen uses only C1
            for b_idx in range(4):
                coeff = gamma[b_idx]
                if coeff != 0:
                    offset = block_offsets[b_idx]
                    for idx in range(4):
                        W_full[offset + idx] += coeff * W_sub[s][idx]
                        # Note: C2 part is 0 for this step
            
            # Assign to U, V, W
            for i in range(16):
                U[i][col] = U_full[i]
            for i in range(20):
                V[i][col] = V_full[i]
            for i in range(20):
                W[i][col] = W_full[i]
            
            col += 1
            
    # Part 2: 4x4x1 multiplication (rank 16)
    # C2 = A * B2
    # A is 4x4 (indices 0-15)
    # B2 is 4x1 (indices 16-19 in V)
    # C2 is 4x1 (indices 16-19 in W)
    
    for i in range(4):
        for j in range(4):
            # U index: i*4 + j
            U[i * 4 + j][col] = 1
            # V index: 16 + j
            V[16 + j][col] = 1
            # W index: 16 + i
            W[16 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

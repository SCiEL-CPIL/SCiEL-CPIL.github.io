"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Indices in 2x2 matrix: 11, 12, 21, 22
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Coefficients for U (X), V (Y), W (Z)
    # Order of elements in 2x2 block: 11, 12, 21, 22
    
    # M1
    U_M1 = [1, 0, 0, 1]
    V_M1 = [1, 0, 0, 1]
    W_M1 = [1, 0, 0, 0] # Z11 += 1
    W_M1[3] += 1        # Z22 += 1
    
    # M2
    U_M2 = [0, 0, 1, 1]
    V_M2 = [1, 0, 0, 0]
    W_M2 = [0, 0, 1, 0] # Z21 += 1
    W_M2[3] += -1       # Z22 += -1
    
    # M3
    U_M3 = [1, 0, 0, 0]
    V_M3 = [0, 1, 0, -1]
    W_M3 = [0, 1, 0, 0] # Z12 += 1
    W_M3[3] += 1        # Z22 += 1
    
    # M4
    U_M4 = [0, 0, 0, 1]
    V_M4 = [0, 0, 1, -1]
    W_M4 = [1, 0, 1, 0] # Z11 += 1, Z21 += 1
    
    # M5
    U_M5 = [1, 1, 0, 0]
    V_M5 = [0, 0, 0, 1]
    W_M5 = [0, 1, 0, 0] # Z12 += 1
    W_M5[0] += -1       # Z11 += -1
    
    # M6
    U_M6 = [-1, 0, 1, 0]
    V_M6 = [1, 1, 0, 0]
    W_M6 = [0, 0, 0, 1] # Z22 += 1
    
    # M7
    U_M7 = [0, 1, 0, -1]
    V_M7 = [0, 0, 1, 1]
    W_M7 = [1, 0, 0, 0] # Z11 += 1
    
    U_coeffs = [U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7]
    V_coeffs = [V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7]
    W_coeffs = [W_M1, W_M2, W_M3, W_M4, W_M5, W_M6, W_M7]
    
    # Define the 4 blocks
    # Each block has:
    # - 4 indices for U (X)
    # - 4 indices for V (Y)
    # - 4 indices for W (Z)
    
    blocks = [
        # Block 0: A1 * B11 -> C1
        {
            'U_idx': [0, 1, 4, 5],   # A[0,0], A[0,1], A[1,0], A[1,1]
            'V_idx': [0, 1, 4, 5],   # B[0,0], B[0,1], B[1,0], B[1,1]
            'W_idx': [0, 1, 2, 3]    # C[0,0], C[0,1], C[1,0], C[1,1]
        },
        # Block 1: A2 * B21 -> C1
        {
            'U_idx': [2, 3, 6, 7],   # A[0,2], A[0,3], A[1,2], A[1,3]
            'V_idx': [8, 9, 12, 13], # B[2,0], B[2,1], B[3,0], B[3,1]
            'W_idx': [0, 1, 2, 3]    # C[0,0], C[0,1], C[1,0], C[1,1]
        },
        # Block 2: A1 * B12 -> C2
        {
            'U_idx': [0, 1, 4, 5],   # A[0,0], A[0,1], A[1,0], A[1,1]
            'V_idx': [2, 3, 6, 7],   # B[0,2], B[0,3], B[1,2], B[1,3]
            'W_idx': [4, 5, 6, 7]    # C[0,2], C[0,3], C[1,2], C[1,3]
        },
        # Block 3: A2 * B22 -> C2
        {
            'U_idx': [2, 3, 6, 7],   # A[0,2], A[0,3], A[1,2], A[1,3]
            'V_idx': [10, 11, 14, 15],# B[2,2], B[2,3], B[3,2], B[3,3]
            'W_idx': [4, 5, 6, 7]    # C[0,2], C[0,3], C[1,2], C[1,3]
        }
    ]
    
    for block in blocks:
        u_idx = block['U_idx']
        v_idx = block['V_idx']
        w_idx = block['W_idx']
        
        for s in range(7):
            col = blocks.index(block) * 7 + s
            # Add coefficients to U
            for local_idx in range(4):
                coeff = U_coeffs[s][local_idx]
                if coeff != 0:
                    U[u_idx[local_idx]][col] += coeff
            
            # Add coefficients to V
            for local_idx in range(4):
                coeff = V_coeffs[s][local_idx]
                if coeff != 0:
                    V[v_idx[local_idx]][col] += coeff
            
            # Add coefficients to W
            for local_idx in range(4):
                coeff = W_coeffs[s][local_idx]
                if coeff != 0:
                    W[w_idx[local_idx]][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

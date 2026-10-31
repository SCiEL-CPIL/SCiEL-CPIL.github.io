"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # U, V, W are 7x4 matrices (flattened 2x2)
    # Index: I*2 + J for U, J*2 + K for V, K*2 + I for W
    
    U_block = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: A10, A00
        [0, 1, 0, -1] # M7: A01, A11
    ]
    
    V_block = [
        [1, 0, 1, 0], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, B11
        [-1, 0, 1, 0],# M4: B10, B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 1, 1, 0]  # M7: B10, B11
    ]
    
    W_block = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Inner decomposition is identical
    U_inner = U_block
    V_inner = V_block
    W_inner = W_block
    
    rank = 49
    n, m, p = 4, 4, 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            # Iterate over all indices I, J, K, i, j, k in {0, 1}
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    # U index
                                    u_idx = (I * 2 + i) * 4 + (J * 2 + j)
                                    u_val = U_block[r][I * 2 + J] * U_inner[s][i * 2 + j]
                                    U[u_idx][col] += u_val
                                    
                                    # V index
                                    v_idx = (J * 2 + j) * 4 + (K * 2 + k)
                                    v_val = V_block[r][J * 2 + K] * V_inner[s][j * 2 + k]
                                    V[v_idx][col] += v_val
                                    
                                    # W index
                                    w_idx = (I * 2 + i) * 4 + (K * 2 + k)
                                    w_val = W_block[r][K * 2 + I] * W_inner[s][k * 2 + i]
                                    W[w_idx][col] += w_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

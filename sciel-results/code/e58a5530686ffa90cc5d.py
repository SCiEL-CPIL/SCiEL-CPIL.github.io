"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    # 8 independent 2x2x2 blocks, each using Strassen (rank 7)
    # Total rank = 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen 2x2x2 decomposition
    # U contributions for A00,A01,A10,A11 (indices 0,1,2,3)
    U_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    # V contributions for B00,B01,B10,B11
    V_strassen = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [0, 1, -1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 1, 0],   # M6
        [0, 1, 0, 1],   # M7
    ]
    # W contributions for C00,C01,C10,C11
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00+=M1, C11+=M1
        [0, 0, 1, -1],  # M2: C10+=M2, C11-=M2
        [0, 1, 0, 1],   # M3: C01+=M3, C11+=M3
        [1, 0, 1, 0],   # M4: C00+=M4, C10+=M4
        [-1, 1, 0, 0],  # M5: C00-=M5, C01+=M5
        [0, 0, 0, 1],   # M6: C11+=M6
        [1, 0, 0, 0],   # M7: C00+=M7
    ]
    
    for g in range(8):  # 8 blocks
        for s in range(7):  # 7 Strassen terms
            col = g * 7 + s
            # U: same indices 0-3 for all blocks
            for idx in range(4):
                U[idx][col] = U_strassen[s][idx]
            # V: B indices for block g: B[j, 2g] and B[j, 2g+1]
            # B00 = B[0,2g] = index 2g, B01 = B[0,2g+1] = index 2g+1
            # B10 = B[1,2g] = index 16+2g, B11 = B[1,2g+1] = index 16+2g+1
            b_idx = [2*g, 2*g+1, 16+2*g, 16+2*g+1]
            for idx in range(4):
                V[b_idx[idx]][col] = V_strassen[s][idx]
            # W: C indices for block g: C[i, 2g] and C[i, 2g+1]
            # W index = k*n+i where k is column, i is row
            # C00 = C[0,2g] -> W[2g*2+0] = W[4g]
            # C01 = C[0,2g+1] -> W[(2g+1)*2+0] = W[4g+2]
            # C10 = C[1,2g] -> W[2g*2+1] = W[4g+1]
            # C11 = C[1,2g+1] -> W[(2g+1)*2+1] = W[4g+3]
            c_idx = [4*g, 4*g+2, 4*g+1, 4*g+3]  # C00, C01, C10, C11
            for idx in range(4):
                W[c_idx[idx]][col] = W_strassen[s][idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

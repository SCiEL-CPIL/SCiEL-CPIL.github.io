"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_strassen = [
        [1,  1,  1,  0,  1, -1,  1], # i=0, j=0 (a11)
        [0,  0,  0,  0,  1,  0,  1], # i=0, j=1 (a12)
        [0,  1,  0,  0,  0,  1,  0], # i=1, j=0 (a21)
        [1,  1,  0,  1,  0,  0, -1]  # i=1, j=1 (a22)
    ]
    
    V_strassen = [
        [1,  1,  0, -1, 0,  1,  0], # j=0, k=0 (b11)
        [0,  0,  1,  0,  0,  1,  0], # j=0, k=1 (b12)
        [0,  0,  0,  1,  0,  0,  1], # j=1, k=0 (b21)
        [1,  0, -1, 0,  1,  0,  1]  # j=1, k=1 (b22)
    ]
    
    W_strassen = [
        [1,  0,  0,  1, -1, 0,  1], # k=0, i=0 (c11)
        [0,  1,  0,  1,  0,  0,  0], # k=0, i=1 (c21)
        [0,  0,  1,  0,  1,  0,  0], # k=1, i=0 (c12)
        [1, -1,  1,  0,  0,  1,  0]  # k=1, i=1 (c22)
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U
    for b in range(3):
        for r in range(7):
            for i in range(4):
                U[i][b * 7 + r] = U_strassen[i][r]
                
    # Fill V and W
    for b in range(3):
        k_offset = b * 2
        for r in range(7):
            # V
            for j in range(2):
                for k in range(2):
                    v_idx = j * p + (k + k_offset)
                    v_val = V_strassen[j * 2 + k][r]
                    V[v_idx][b * 7 + r] = v_val
            # W
            for k in range(2):
                for i in range(2):
                    w_idx = (k + k_offset) * n + i
                    w_val = W_strassen[k * 2 + i][r]
                    W[w_idx][b * 7 + r] = w_val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

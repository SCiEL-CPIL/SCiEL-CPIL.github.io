"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2x2
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V local indices: 0:e, 1:f, 2:g, 3:h
    # W local indices: 0:m11, 1:m21, 2:m12, 3:m22
    
    strassen_u = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [0, 0, 1, -1], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [0, -1, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    strassen_w = [
        [1, 0, 0, 1],  # P1
        [0, 1, 0, -1], # P2
        [0, 0, 1, 1],  # P3
        [1, 1, 0, 0],  # P4
        [-1, 0, 1, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 0, 0, 0]   # P7
    ]
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        K = block * 2
        for s in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    idx_u = i * m + j
                    U[idx_u][col] = strassen_u[s][i * 2 + j]
            
            # V
            for j in range(2):
                for k_local in range(2):
                    k = K + k_local
                    idx_v = j * p + k
                    V[idx_v][col] = strassen_v[s][j * 2 + k_local]
            
            # W
            for k_local in range(2):
                k = K + k_local
                for i in range(2):
                    idx_w = k * n + i
                    W[idx_w][col] = strassen_w[s][k_local * 2 + i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

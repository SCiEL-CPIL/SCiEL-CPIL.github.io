import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition vectors
    # u, v, w are length 4
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0]
    ]
    v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    # 49 terms for 4x4 by 4x4 (first 4 cols of B, first 4 cols of C)
    for a in range(7):
        for b in range(7):
            # U term: u_a tensor u_b
            # u_a is length 4 (indices i1, j1 for 2x2 block)
            # u_b is length 4 (indices i2, j2 for 2x2 block)
            # Total index in 4x4: (i1*2 + i2) * 4 + (j1*2 + j2)
            for i1 in range(2):
                for j1 in range(2):
                    val_a = u[a][i1*2 + j1]
                    if val_a == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val_b = u[b][i2*2 + j2]
                            if val_b == 0: continue
                            idx_U = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[idx_U][col] += val_a * val_b
                            
            # V term: v_a tensor v_b
            # Only first 4 cols of B (indices 0..3)
            for j1 in range(2):
                for k1 in range(2):
                    val_a = v[a][j1*2 + k1]
                    if val_a == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            val_b = v[b][j2*2 + k2]
                            if val_b == 0: continue
                            idx_V = (j1 * 2 + j2) * 5 + (k1 * 2 + k2)
                            V[idx_V][col] += val_a * val_b
                            
            # W term: w_a tensor w_b
            # Only first 4 cols of C (indices 0..3)
            for k1 in range(2):
                for i1 in range(2):
                    val_a = w[a][k1*2 + i1]
                    if val_a == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            val_b = w[b][k2*2 + i2]
                            if val_b == 0: continue
                            idx_W = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W[idx_W][col] += val_a * val_b
                            
            col += 1
            
    # 16 terms for 4x4 by 4x1 (last col of B, last col of C)
    # C[4, i] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            idx_U = i * 4 + j
            idx_V = j * 5 + 4
            idx_W = 4 * 4 + i
            U[idx_U][col] = 1
            V[idx_V][col] = 1
            W[idx_W][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

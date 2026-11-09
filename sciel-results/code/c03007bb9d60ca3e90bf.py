import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds margin before 150s limit
    
    n, m, p = 2, 4, 5
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 7x4 matrices (flattened 2x2 blocks)
    # Indices: i, j in {0,1}
    # U[i*2+j]
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    # The known rank 36 decomposition:
    # 4 blocks of 2x2x2 (Strassen) for the 2x4x4 part = 28
    # Schoolbook for the 2x4x1 part (last column of B) = 8
    # Total = 36
    
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Decompose 2x4x4 part using 4 Strassen blocks
    # Precompute Strassen block contributions for efficiency
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                # U part: 2x2 block at rows 0-1, cols j0-j0+1
                U[0 * m + j0][col] = strassen_U[r][0]
                U[0 * m + j0 + 1][col] = strassen_U[r][1]
                U[1 * m + j0][col] = strassen_U[r][2]
                U[1 * m + j0 + 1][col] = strassen_U[r][3]
                
                # V part: 2x2 block at rows j0-j0+1, cols k0-k0+1
                V[j0 * p + k0][col] = strassen_V[r][0]
                V[j0 * p + k0 + 1][col] = strassen_V[r][1]
                V[(j0 + 1) * p + k0][col] = strassen_V[r][2]
                V[(j0 + 1) * p + k0 + 1][col] = strassen_V[r][3]
                
                # W part: 2x2 block at rows k0-k0+1, cols 0-1
                W[k0 * n + 0][col] = strassen_W[r][0]
                W[k0 * n + 1][col] = strassen_W[r][1]
                W[(k0 + 1) * n + 0][col] = strassen_W[r][2]
                W[(k0 + 1) * n + 1][col] = strassen_W[r][3]
                
                col += 1
    
    # Decompose 2x4x1 part (last column of B) using schoolbook
    # C[:, 4] = A * B[:, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

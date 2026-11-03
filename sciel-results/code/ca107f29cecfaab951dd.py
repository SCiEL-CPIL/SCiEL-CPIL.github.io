import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # Rank 16
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # Rank 28 (4 blocks of Strassen 2x2x2)
    for block in range(4):
        # Block indices
        b00_idx = 1 * 8 + block * 2
        b01_idx = 1 * 8 + block * 2 + 1
        b10_idx = 2 * 8 + block * 2
        b11_idx = 2 * 8 + block * 2 + 1
        
        c00_idx = (block * 2) * 2 + 0
        c01_idx = (block * 2 + 1) * 2 + 0
        c10_idx = (block * 2) * 2 + 1
        c11_idx = (block * 2 + 1) * 2 + 1
        
        # Strassen terms
        # 1. (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[b00_idx][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # 2. (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[b00_idx][col] = 1
        W[c10_idx][col] = 1
        W[c11_idx][col] = -1
        col += 1
        
        # 3. a00(b01-b11)
        U[1][col] = 1
        V[b01_idx][col] = 1
        V[b11_idx][col] = -1
        W[c01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # 4. a11(b10-b00)
        U[5][col] = 1
        V[b10_idx][col] = 1
        V[b00_idx][col] = -1
        W[c00_idx][col] = 1
        W[c10_idx][col] = 1
        col += 1
        
        # 5. (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = -1
        W[c01_idx][col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[b00_idx][col] = 1
        V[b01_idx][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[b10_idx][col] = 1
        V[b11_idx][col] = 1
        W[c00_idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

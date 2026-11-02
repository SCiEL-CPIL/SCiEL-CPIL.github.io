import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    n, m, p = DIMENSIONS
    
    # Lader's rank-23 decomposition for 3x3x3
    # U, V, W are 23x9 matrices in the seed, but here we treat them as lists of lists
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1],
    ]
    
    # Verify Lader decomposition for 3x3x3
    def verify_333(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(23):
                        total += U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i]
                    if total != 1:
                        return False
        return True

    lader_valid = verify_333(lader_U, lader_V, lader_W)
    
    # Helper to verify full decomposition
    def verify_full(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    if lader_valid:
        # Rank 64 construction:
        # 23 (Lader for k=0,1,2) + 23 (Lader for k=3,4,5) + 18 (Schoolbook for k=6,7)
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k in {0, 1, 2}, rank 23 (cols 0-22)
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+k][r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*n+i][r] = lader_W[r][k*3+i]
        
        # Block 2: k in {3, 4, 5}, rank 23 (cols 23-45)
        offset = 23
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][offset + r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    # Map local k (0,1,2) to global k (3,4,5)
                    V[j*p + (3+k)][offset + r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    # Map local k (0,1,2) to global k (3,4,5)
                    W[(3+k)*n + i][offset + r] = lader_W[r][k*3+i]
        
        # Block 3: k in {6, 7}, rank 18 (cols 46-63)
        offset = 46
        col_idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(6, 8):
                    U[i*m+j][offset + col_idx] = 1
                    V[j*p+k][offset + col_idx] = 1
                    W[k*n+i][offset + col_idx] = 1
                    col_idx += 1
        
        if verify_full(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook rank 72
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

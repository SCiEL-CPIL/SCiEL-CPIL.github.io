import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 by 5x5 matrix.
    Uses partitioning into 2x2x2 (Strassen) blocks.
    """
    n, m, p = 4, 5, 5
    deadline = time.time() + 110.0
    
    # Strategy: Use 4 disjoint 2x2x2 Strassen blocks.
    # Blocks:
    # 1. I=[0,1], J=[0,1], K=[0,1]
    # 2. I=[2,3], J=[2,3], K=[2,3]
    # 3. I=[0,1], J=[2,3], K=[2,3]
    # 4. I=[2,3], J=[0,1], K=[0,1]
    # Remaining terms are handled by schoolbook.
    
    # Total rank: 4 * 7 (Strassen) + (100 - 4 * 8) (Schoolbook) = 28 + 68 = 96.
    
    rank = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [0, 1])
    ]
    
    skip_set = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    skip_set.add((i, j, k))
    
    col = 0
    # Add schoolbook terms for non-block indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Add Strassen terms for blocks
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        W[k2 * n + i2][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1][col] += 1
        V[j1 * p + k2][col] += 1
        V[j2 * p + k2][col] += -1
        W[k2 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2][col] += 1
        V[j2 * p + k1][col] += 1
        V[j1 * p + k1][col] += -1
        W[k1 * n + i1][col] += 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += -1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] += -1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

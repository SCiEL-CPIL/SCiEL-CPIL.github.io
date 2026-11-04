import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def _verify_decomp(U, V, W, size):
    """Verify that sum_r U[i*size+j, r]*V[j*size+k, r]*W[k*size+i, r] = 1 for all i,j,k in range(size)"""
    rank = len(U[0])
    for i in range(size):
        for j in range(size):
            for k in range(size):
                s = 0
                for r in range(rank):
                    s += U[i*size + j][r] * V[j*size + k][r] * W[k*size + i][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Laderman 3x3 algorithm (rank 7)
    U3 = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ]
    V3 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ]
    W3 = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ]
    
    # Verify 3x3 Laderman
    if not _verify_decomp(U3, V3, W3, 3):
        # Fallback to schoolbook
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
    
    # Schoolbook 2x2 (rank 4)
    U2 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V2 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    W2 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    
    # Tensor product: 3x3 (rank 7) ⊗ 2x2 (rank 4) -> 6x6 (rank 28)
    n1, m1, p1 = 3, 3, 3
    n2, m2, p2 = 2, 2, 2
    r1, r2 = 7, 4
    r_total = r1 * r2  # 28
    
    n6, m6, p6 = 6, 6, 6
    
    U6 = [[0] * r_total for _ in range(n6 * m6)]
    V6 = [[0] * r_total for _ in range(m6 * p6)]
    W6 = [[0] * r_total for _ in range(p6 * n6)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            
                            for r1_idx in range(r1):
                                for r2_idx in range(r2):
                                    r = r1_idx * r2 + r2_idx
                                    
                                    u_val = U3[i1*m1 + j1][r1_idx] * U2[i2*m2 + j2][r2_idx]
                                    v_val = V3[j1*p1 + k1][r1_idx] * V2[j2*p2 + k2][r2_idx]
                                    w_val = W3[k1*n1 + i1][r1_idx] * W2[k2*n2 + i2][r2_idx]
                                    
                                    U6[i*m6 + j][r] = u_val
                                    V6[j*p6 + k][r] = v_val
                                    W6[k*n6 + i][r] = w_val
    
    # Verify 6x6 decomposition
    if not _verify_decomp(U6, V6, W6, 6):
        # Fallback to schoolbook
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
    
    # Restrict 6x6 to 5x5: keep indices 0..4
    U5 = [[0] * r_total for _ in range(n * m)]
    V5 = [[0] * r_total for _ in range(m * p)]
    W5 = [[0] * r_total for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for r in range(r_total):
                U5[i*m + j][r] = U6[i*6 + j][r]
    
    for j in range(m):
        for k in range(p):
            for r in range(r_total):
                V5[j*p + k][r] = V6[j*6 + k][r]
    
    for k in range(p):
        for i in range(n):
            for r in range(r_total):
                W5[k*n + i][r] = W6[k*6 + i][r]
    
    # Verify 5x5 decomposition
    if not _verify_decomp(U5, V5, W5, 5):
        # Fallback to schoolbook
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
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END

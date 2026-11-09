"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 58: 8 Strassen blocks (7 each) + 2 tail terms (1 each)
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        for t in range(7):
            r = col + t
            # U assignments
            if t == 0:
                U[(i_base + 0) * m + (j_base + 0)][r] = 1
                U[(i_base + 1) * m + (j_base + 1)][r] = 1
            elif t == 1:
                U[(i_base + 1) * m + (j_base + 0)][r] = 1
                U[(i_base + 1) * m + (j_base + 1)][r] = 1
            elif t == 2:
                U[(i_base + 0) * m + (j_base + 0)][r] = 1
            elif t == 3:
                U[(i_base + 1) * m + (j_base + 1)][r] = 1
            elif t == 4:
                U[(i_base + 0) * m + (j_base + 0)][r] = 1
                U[(i_base + 1) * m + (j_base + 0)][r] = 1
            elif t == 5:
                U[(i_base + 1) * m + (j_base + 0)][r] = 1
                U[(i_base + 0) * m + (j_base + 0)][r] = -1
            elif t == 6:
                U[(i_base + 0) * m + (j_base + 1)][r] = 1
                U[(i_base + 1) * m + (j_base + 1)][r] = -1
            
            # V assignments
            if t == 0:
                V[(j_base + 0) * p + (k_base + 0)][r] = 1
                V[(j_base + 1) * p + (k_base + 1)][r] = 1
            elif t == 1:
                V[(j_base + 0) * p + (k_base + 0)][r] = 1
            elif t == 2:
                V[(j_base + 1) * p + (k_base + 0)][r] = 1
                V[(j_base + 1) * p + (k_base + 1)][r] = -1
            elif t == 3:
                V[(j_base + 0) * p + (k_base + 1)][r] = 1
                V[(j_base + 0) * p + (k_base + 0)][r] = -1
            elif t == 4:
                V[(j_base + 1) * p + (k_base + 1)][r] = 1
            elif t == 5:
                V[(j_base + 0) * p + (k_base + 0)][r] = 1
                V[(j_base + 0) * p + (k_base + 1)][r] = 1
            elif t == 6:
                V[(j_base + 1) * p + (k_base + 1)][r] = 1
                V[(j_base + 0) * p + (k_base + 1)][r] = 1
            
            # W assignments
            if t == 0:
                W[(k_base + 0) * n + (i_base + 0)][r] = 1
                W[(k_base + 1) * n + (i_base + 1)][r] = 1
            elif t == 1:
                W[(k_base + 0) * n + (i_base + 1)][r] = 1
                W[(k_base + 0) * n + (i_base + 0)][r] = -1
            elif t == 2:
                W[(k_base + 0) * n + (i_base + 0)][r] = 1
            elif t == 3:
                W[(k_base + 0) * n + (i_base + 0)][r] = -1
                W[(k_base + 1) * n + (i_base + 0)][r] = 1
            elif t == 4:
                W[(k_base + 1) * n + (i_base + 0)][r] = 1
                W[(k_base + 1) * n + (i_base + 1)][r] = -1
            elif t == 5:
                W[(k_base + 1) * n + (i_base + 1)][r] = 1
            elif t == 6:
                W[(k_base + 0) * n + (i_base + 0)][r] = 1
                W[(k_base + 1) * n + (i_base + 1)][r] = -1
        
        col += 7

    # 8 Strassen blocks covering i in {0,1} x {2,3}, j in {0,1} x {2,3}, k in {0,1} x {2,3}
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    add_strassen(2, 0, 0)
    add_strassen(2, 2, 0)
    add_strassen(2, 0, 2)
    add_strassen(2, 2, 2)
    
    # Tail 1: j=4 (all i, k)
    r = col
    for i in range(n):
        U[i * m + 4][r] = 1
    for k in range(p):
        V[4 * p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][r] = 1
    col += 1
    
    # Tail 2: k=4 (all i, j in 0..3)
    r = col
    for i in range(n):
        for j in range(4):
            U[i * m + j][r] = 1
    for j in range(4):
        V[j * p + 4][r] = 1
    for i in range(n):
        W[4 * n + i][r] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

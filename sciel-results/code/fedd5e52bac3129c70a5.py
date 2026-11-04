import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Part 1: i=2 (1x4x7 schoolbook)
    # i=2, j in 0..3, k in 0..6
    for j in range(m):
        for k in range(p):
            U[2 * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + 2][c] = 1
            c += 1
            
    # Part 2: i in 0..1 (2x4x7)
    # Split j into 0..1 and 2..3
    for j_base in [0, 2]:
        # Split k into 0..1, 2..3, 4..5 (Strassen) and 6 (Schoolbook)
        for k_base in [0, 2, 4]:
            # Strassen 2x2x2
            # Local i' in 0..1, j' in 0..1, k' in 0..1
            # Global i = i', j = j_base + j', k = k_base + k'
            
            # m1: (x00+x11)(y00+y11) -> z00, z11
            U[i_base + 0][c] = 1 # i_base is 0 or 1? No, i is 0 or 1 globally.
            # Wait, i is 0 or 1. j_base is 0 or 2.
            # For i=0:
            U[0 * m + (j_base + 0)][c] = 1
            U[1 * m + (j_base + 1)][c] = 1
            V[(j_base + 0) * p + (k_base + 0)][c] = 1
            V[(j_base + 1) * p + (k_base + 1)][c] = 1
            W[(k_base + 0) * n + 0][c] = 1
            W[(k_base + 1) * n + 1][c] = 1
            c += 1
            
            # m2: (x10+x11)y00 -> z10, z11
            U[1 * m + (j_base + 0)][c] = 1
            U[1 * m + (j_base + 1)][c] = 1
            V[(j_base + 0) * p + (k_base + 0)][c] = 1
            W[(k_base + 0) * n + 1][c] = 1
            W[(k_base + 1) * n + 1][c] = -1
            c += 1
            
            # m3: x00(y01-y11) -> z01, z11
            U[0 * m + (j_base + 0)][c] = 1
            V[(j_base + 0) * p + (k_base + 1)][c] = 1
            V[(j_base + 1) * p + (k_base + 1)][c] = -1
            W[(k_base + 1) * n + 0][c] = 1
            W[(k_base + 1) * n + 1][c] = 1
            c += 1
            
            # m4: x11(y10-y00) -> z00, z10
            U[1 * m + (j_base + 1)][c] = 1
            V[(j_base + 1) * p + (k_base + 0)][c] = 1
            V[(j_base + 0) * p + (k_base + 0)][c] = -1
            W[(k_base + 0) * n + 0][c] = 1
            W[(k_base + 0) * n + 1][c] = 1
            c += 1
            
            # m5: (x00+x01)y11 -> z00, z01
            U[0 * m + (j_base + 0)][c] = 1
            U[0 * m + (j_base + 1)][c] = 1
            V[(j_base + 1) * p + (k_base + 1)][c] = 1
            W[(k_base + 0) * n + 0][c] = -1
            W[(k_base + 1) * n + 0][c] = 1
            c += 1
            
            # m6: (x10-x00)(y00+y01) -> z11
            U[1 * m + (j_base + 0)][c] = 1
            U[0 * m + (j_base + 0)][c] = -1
            V[(j_base + 0) * p + (k_base + 0)][c] = 1
            V[(j_base + 0) * p + (k_base + 1)][c] = 1
            W[(k_base + 1) * n + 1][c] = 1
            c += 1
            
            # m7: (x01-x11)(y10+y11) -> z00, z11
            U[0 * m + (j_base + 1)][c] = 1
            U[1 * m + (j_base + 1)][c] = -1
            V[(j_base + 1) * p + (k_base + 0)][c] = 1
            V[(j_base + 1) * p + (k_base + 1)][c] = 1
            W[(k_base + 0) * n + 0][c] = 1
            W[(k_base + 1) * n + 1][c] = -1
            c += 1
            
        # k_base = 6 (2x2x1 schoolbook)
        # x00 y00 -> z00
        U[0 * m + (j_base + 0)][c] = 1
        V[(j_base + 0) * p + 6][c] = 1
        W[6 * n + 0][c] = 1
        c += 1
        
        # x01 y10 -> z00
        U[0 * m + (j_base + 1)][c] = 1
        V[(j_base + 1) * p + 6][c] = 1
        W[6 * n + 0][c] = 1
        c += 1
        
        # x10 y00 -> z10
        U[1 * m + (j_base + 0)][c] = 1
        V[(j_base + 0) * p + 6][c] = 1
        W[6 * n + 1][c] = 1
        c += 1
        
        # x11 y10 -> z10
        U[1 * m + (j_base + 1)][c] = 1
        V[(j_base + 1) * p + 6][c] = 1
        W[6 * n + 1][c] = 1
        c += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

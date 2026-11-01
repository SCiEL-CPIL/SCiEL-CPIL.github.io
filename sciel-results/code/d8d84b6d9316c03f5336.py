import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 52 decomposition.
    # Part 1: 2x4x4 using 4 Strassen 2x2x2 blocks (rank 7 each) -> 28
    # Part 2: 2x4x3 using schoolbook (rank 24) -> 24
    # Total rank 52.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # j_offset: offset for j indices (0 or 2)
    # k_offset: offset for k indices (0, 2, 4, 6)
    def add_strassen_block(j_off, k_off):
        nonlocal col
        # Strassen formulas for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # Indices in A: i * m + j. Here i in {0,1}, j in {j_off, j_off+1}
        # Indices in B: j * p + k. Here j in {j_off, j_off+1}, k in {k_off, k_off+1}
        # Indices in C: k * n + i. Here k in {k_off, k_off+1}, i in {0,1}
        
        # M1
        U[0 * m + j_off][col] = 1
        U[1 * m + (j_off + 1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[k_off * n + 0][col] = 1
        W[(k_off + 1) * n + 0][col] = 1
        W[k_off * n + 1][col] = 1
        W[(k_off + 1) * n + 1][col] = 1
        col += 1
        
        # M2
        U[1 * m + j_off][col] = 1
        U[1 * m + (j_off + 1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + 1][col] = 1
        col += 1
        
        # M3
        U[0 * m + j_off][col] = 1
        V[(j_off + 1) * p + k_off][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = -1
        W[k_off * n + 0][col] = 1
        col += 1
        
        # M4
        U[1 * m + (j_off + 1)][col] = 1
        V[j_off * p + (k_off + 1)][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + 1][col] = 1
        col += 1
        
        # M5
        U[0 * m + j_off][col] = 1
        U[0 * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[k_off * n + 0][col] = -1
        W[(k_off + 1) * n + 0][col] = -1
        col += 1
        
        # M6
        U[1 * m + j_off][col] = 1
        U[0 * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[(j_off + 1) * p + k_off][col] = 1
        W[(k_off + 1) * n + 1][col] = 1
        col += 1
        
        # M7
        U[0 * m + (j_off + 1)][col] = 1
        U[1 * m + (j_off + 1)][col] = -1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        V[j_off * p + (k_off + 1)][col] = 1
        W[k_off * n + 0][col] = 1
        col += 1

    # Apply Strassen for the 4x4 part (k in 0..3)
    # j in 0..1, k in 0..1
    add_strassen_block(0, 0)
    # j in 0..1, k in 2..3
    add_strassen_block(0, 2)
    # j in 2..3, k in 0..1
    add_strassen_block(2, 0)
    # j in 2..3, k in 2..3
    add_strassen_block(2, 2)
    
    # Schoolbook for the 4x3 part (k in 4..6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(j_off, k_off, col_start):
        # 7 steps of Strassen for 2x2x2
        # Indices:
        # A: i in {0,1}, j in {j_off, j_off+1}
        # B: j in {j_off, j_off+1}, k in {k_off, k_off+1}
        # C: i in {0,1}, k in {k_off, k_off+1}
        
        # P1 = (A00 + A11)(B00 + B11)
        c = col_start
        U[0 * m + j_off][c] += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[j_off * p + k_off][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1
        
        # P2 = (A10 + A11) B00
        U[1 * m + j_off][c] += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[j_off * p + k_off][c] += 1
        W[k_off * n + 1][c] += 1
        W[(k_off + 1) * n + 1][c] -= 1
        c += 1
        
        # P3 = A00 (B01 - B11)
        U[0 * m + j_off][c] += 1
        V[j_off * p + (k_off + 1)][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] -= 1
        W[(k_off + 1) * n + 0][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1
        
        # P4 = A11 (B10 - B00)
        U[1 * m + (j_off + 1)][c] += 1
        V[(j_off + 1) * p + k_off][c] += 1
        V[j_off * p + k_off][c] -= 1
        W[k_off * n + 0][c] += 1
        W[k_off * n + 1][c] += 1
        c += 1
        
        # P5 = (A00 + A01) B11
        U[0 * m + j_off][c] += 1
        U[0 * m + (j_off + 1)][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] -= 1
        W[(k_off + 1) * n + 0][c] += 1
        c += 1
        
        # P6 = (A10 - A00)(B00 + B01)
        U[1 * m + j_off][c] += 1
        U[0 * m + j_off][c] -= 1
        V[j_off * p + k_off][c] += 1
        V[j_off * p + (k_off + 1)][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1
        
        # P7 = (A01 - A11)(B10 + B11)
        U[0 * m + (j_off + 1)][c] += 1
        U[1 * m + (j_off + 1)][c] -= 1
        V[(j_off + 1) * p + k_off][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] += 1

    # Block 1: 2x4x4 using 4x Strassen
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 7)
    add_strassen(2, 0, 14)
    add_strassen(2, 2, 21)
    
    # Block 2: A_1 B_14 (j=4, k in 0..3)
    col = 28
    for i in range(2):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 3: A_4 B_41 (j in 0..3, k=4)
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Block 4: A_1 B_11 (j=4, k=4)
    for i in range(2):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

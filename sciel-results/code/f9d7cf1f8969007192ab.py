import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A is 2x4, B is 4x5.
    # Split B into B1 (4x4) and B2 (4x1).
    # C = A B1 + A B2.
    # A B1: A (2x4) * B1 (4x4).
    # Split A into A1 (2x2) and A2 (2x2).
    # B1 into B11, B12, B21, B22 (2x2 each).
    # C_left = A1 B11 + A2 B21
    # C_right = A1 B12 + A2 B22
    # Each 2x2 * 2x2 uses Strassen (rank 7).
    # Total 4 * 7 = 28 mults for A B1.
    # A B2: A (2x4) * B2 (4x1).
    # This is 2 dot products of length 4. Rank 8.
    # Total rank 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    # Inputs: A indices (i,j), B indices (k,l)
    # Output: C indices (i,l)
    # Strassen formulas for C = A * B
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = (A11 + A12)B22
    # M4 = (A22 - A11)(B11 + B12)
    # M5 = (A11 + A21)(B22 + B21)
    # M6 = (A22 + A12)(B11 + B12)
    # M7 = (A11 + A12 - A21 - A22)(B22 + B21 - B11 - B12)
    # C11 = M1 + M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M5
    # C22 = M1 - M4 + M2 + M3
    
    col = 0
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows, a_cols: indices of A block (2x2)
        # b_rows, b_cols: indices of B block (2x2)
        # c_rows, c_cols: indices of C block (2x2)
        
        # A indices
        A11 = (a_rows[0], a_cols[0])
        A12 = (a_rows[0], a_cols[1])
        A21 = (a_rows[1], a_cols[0])
        A22 = (a_rows[1], a_cols[1])
        
        # B indices
        B11 = (b_rows[0], b_cols[0])
        B12 = (b_rows[0], b_cols[1])
        B21 = (b_rows[1], b_cols[0])
        B22 = (b_rows[1], b_cols[1])
        
        # C indices
        C11 = (c_rows[0], c_cols[0])
        C12 = (c_rows[0], c_cols[1])
        C21 = (c_rows[1], c_cols[0])
        C22 = (c_rows[1], c_cols[1])
        
        # M1 = (A11 + A22)(B11 + B22)
        U[A11[0] * m + A11[1]][col] += 1
        U[A22[0] * m + A22[1]][col] += 1
        V[B11[0] * p + B11[1]][col] += 1
        V[B22[0] * p + B22[1]][col] += 1
        W[C11[0] * n + C11[1]][col] += 1
        W[C22[0] * n + C22[1]][col] += 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[A21[0] * m + A21[1]][col] += 1
        U[A22[0] * m + A22[1]][col] += 1
        V[B11[0] * p + B11[1]][col] += 1
        W[C21[0] * n + C21[1]][col] += 1
        W[C22[0] * n + C22[1]][col] += 1
        col += 1
        
        # M3 = (A11 + A12)B22
        U[A11[0] * m + A11[1]][col] += 1
        U[A12[0] * m + A12[1]][col] += 1
        V[B22[0] * p + B22[1]][col] += 1
        W[C12[0] * n + C12[1]][col] += 1
        W[C22[0] * n + C22[1]][col] += 1
        col += 1
        
        # M4 = (A22 - A11)(B11 + B12)
        U[A22[0] * m + A22[1]][col] += 1
        U[A11[0] * m + A11[1]][col] -= 1
        V[B11[0] * p + B11[1]][col] += 1
        V[B12[0] * p + B12[1]][col] += 1
        W[C11[0] * n + C11[1]][col] += 1
        W[C22[0] * n + C22[1]][col] -= 1
        col += 1
        
        # M5 = (A11 + A21)(B22 + B21)
        U[A11[0] * m + A11[1]][col] += 1
        U[A21[0] * m + A21[1]][col] += 1
        V[B22[0] * p + B22[1]][col] += 1
        V[B21[0] * p + B21[1]][col] += 1
        W[C12[0] * n + C12[1]][col] += 1
        W[C21[0] * n + C21[1]][col] += 1
        col += 1
        
        # M6 = (A22 + A12)(B11 + B12)
        U[A22[0] * m + A22[1]][col] += 1
        U[A12[0] * m + A12[1]][col] += 1
        V[B11[0] * p + B11[1]][col] += 1
        V[B12[0] * p + B12[1]][col] += 1
        W[C11[0] * n + C11[1]][col] -= 1
        col += 1
        
        # M7 = (A11 + A12 - A21 - A22)(B22 + B21 - B11 - B12)
        U[A11[0] * m + A11[1]][col] += 1
        U[A12[0] * m + A12[1]][col] += 1
        U[A21[0] * m + A21[1]][col] -= 1
        U[A22[0] * m + A22[1]][col] -= 1
        V[B22[0] * p + B22[1]][col] += 1
        V[B21[0] * p + B21[1]][col] += 1
        V[B11[0] * p + B11[1]][col] -= 1
        V[B12[0] * p + B12[1]][col] -= 1
        W[C11[0] * n + C11[1]][col] += 1
        col += 1
        
    # A B1 part
    # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
    # B1 is 4x4. B11 is rows 0,1 cols 0,1. B12 is rows 0,1 cols 2,3.
    # B21 is rows 2,3 cols 0,1. B22 is rows 2,3 cols 2,3.
    
    # C1 = A1 B11 + A2 B21
    # C1 is 2x2, cols 0,1 of C.
    # We compute A1 B11 and add to C1.
    # We compute A2 B21 and add to C1.
    # This requires two separate Strassen decompositions?
    # Yes, because the multiplications are distinct.
    # However, we can just call add_strassen for each term.
    
    # Term 1: A1 * B11
    # A1 rows 0,1 cols 0,1
    # B11 rows 0,1 cols 0,1
    # C1 rows 0,1 cols 0,1
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Term 2: A2 * B21
    # A2 rows 0,1 cols 2,3
    # B21 rows 2,3 cols 0,1
    # C1 rows 0,1 cols 0,1
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # C2 = A1 B12 + A2 B22
    # Term 3: A1 * B12
    # A1 rows 0,1 cols 0,1
    # B12 rows 0,1 cols 2,3
    # C2 rows 0,1 cols 2,3
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Term 4: A2 * B22
    # A2 rows 0,1 cols 2,3
    # B22 rows 2,3 cols 2,3
    # C2 rows 0,1 cols 2,3
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # A B2 part
    # A is 2x4. B2 is 4x1 (col 4 of B).
    # C3 is 2x1 (col 4 of C).
    # C3[0] = A[0,0]B2[0,0] + A[0,1]B2[1,0] + A[0,2]B2[2,0] + A[0,3]B2[3,0]
    # C3[1] = A[1,0]B2[0,0] + A[1,1]B2[1,0] + A[1,2]B2[2,0] + A[1,3]B2[3,0]
    
    for j in range(4):
        # A[0,j] * B2[j,0]
        U[0 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 0][col] = 1
        col += 1
        # A[1,j] * B2[j,0]
        U[1 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

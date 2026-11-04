import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Rank 20
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A is 2x2, B is 2x2 (part of the 3x4 B)
    # C is 2x2
    # Maps A indices (i,j) in 0..1 to global i,j
    # Maps B indices (j,k) in 0..1 to global j, k_offset+j
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        # (A11+A22)(B11+B22)
        U[i_off*3 + 0][col] = 1
        U[i_off*3 + 1][col] = 1
        V[0*4 + k_off][col] = 1
        V[1*4 + k_off][col] = 1
        W[k_off*3 + i_off][col] = 1
        W[k_off*3 + i_off+1][col] = 1
        col += 1
        
        # M2
        # (A21+A22)B11
        U[i_off*3 + 1][col] = 1
        V[0*4 + k_off][col] = 1
        W[k_off*3 + i_off+1][col] = 1
        col += 1
        
        # M3
        # A11(B12-B22)
        U[i_off*3 + 0][col] = 1
        V[0*4 + k_off+1][col] = 1
        V[1*4 + k_off+1][col] = -1
        W[(k_off+1)*3 + i_off][col] = 1
        col += 1
        
        # M4
        # A22(B21-B11)
        U[i_off*3 + 1][col] = 1
        V[1*4 + k_off][col] = 1
        V[0*4 + k_off][col] = -1
        W[k_off*3 + i_off+1][col] = 1
        col += 1
        
        # M5
        # (A11+A12)B22
        U[i_off*3 + 0][col] = 1
        U[i_off*3 + 1][col] = 1
        V[1*4 + k_off+1][col] = 1
        W[(k_off+1)*3 + i_off][col] = 1
        W[(k_off+1)*3 + i_off+1][col] = 1
        col += 1
        
        # M6
        # (A21-A11)(B11+B12)
        U[i_off*3 + 1][col] = 1
        U[i_off*3 + 0][col] = -1
        V[0*4 + k_off][col] = 1
        V[0*4 + k_off+1][col] = 1
        W[k_off*3 + i_off+1][col] = 1
        W[(k_off+1)*3 + i_off+1][col] = 1
        col += 1
        
        # M7
        # (A12-A22)(B21+B22)
        U[i_off*3 + 1][col] = 1
        U[i_off*3 + 0][col] = -1
        V[1*4 + k_off][col] = 1
        V[1*4 + k_off+1][col] = 1
        W[k_off*3 + i_off][col] = 1
        W[(k_off+1)*3 + i_off][col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # i_off=0, j_off=0, k_off=0
    add_strassen(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # i_off=0, j_off=0, k_off=2
    add_strassen(0, 0, 2)
    
    # Term 3: A[0:2, 2] * B[2, 0:4]
    # A is 2x1 (col 2), B is 1x4 (row 2)
    # C is 2x4
    # C[i, k] = A[i, 2] * B[2, k]
    # This is an outer product. Rank 1.
    # We can use a single term?
    # L_i = A[i, 2] for i=0,1.
    # M_k = B[2, k] for k=0..3.
    # W_{k,i} = 1.
    U[0*3 + 2][col] = 1
    U[1*3 + 2][col] = 1
    V[2*4 + 0][col] = 1
    V[2*4 + 1][col] = 1
    V[2*4 + 2][col] = 1
    V[2*4 + 3][col] = 1
    W[0*3 + 0][col] = 1
    W[1*3 + 0][col] = 1
    W[2*3 + 0][col] = 1
    W[3*3 + 0][col] = 1
    W[0*3 + 1][col] = 1
    W[1*3 + 1][col] = 1
    W[2*3 + 1][col] = 1
    W[3*3 + 1][col] = 1
    col += 1
    
    # Term 4: A[2, 0:2] * B[0:2, 0:4]
    # A is 1x2 (row 2), B is 2x4
    # C is 1x4
    # C[k] = A[2, 0]B[0, k] + A[2, 1]B[1, k]
    # This requires 4 multiplications (one for each k).
    for k in range(4):
        U[2*3 + 0][col] = 1
        V[0*4 + k][col] = 1
        W[k*3 + 2][col] = 1
        
        U[2*3 + 1][col] = 1
        V[1*4 + k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1
        
    # Wait, I used 4 terms here.
    # Term 5: A[2, 2] * B[2, 0:4]
    # A is 1x1, B is 1x4
    # C[k] = A[2, 2] B[2, k]
    # Rank 1.
    U[2*3 + 2][col] = 1
    V[2*4 + 0][col] = 1
    V[2*4 + 1][col] = 1
    V[2*4 + 2][col] = 1
    V[2*4 + 3][col] = 1
    W[0*3 + 2][col] = 1
    W[1*3 + 2][col] = 1
    W[2*3 + 2][col] = 1
    W[3*3 + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

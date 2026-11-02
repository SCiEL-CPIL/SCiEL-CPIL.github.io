import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    
    # We will use rank 36
    R = 36
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i_offset, j_offset, k_offset are the starting indices for A, B, C blocks
    # A block is at A[i_offset:i_offset+2, j_offset:j_offset+2]
    # B block is at B[j_offset:j_offset+2, k_offset:k_offset+2]
    # C block is at C[k_offset:k_offset+2, i_offset:i_offset+2]
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen terms
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        # M2 = (c+d)e     -> C10 += 1, C11 -= 1
        # M3 = a(f-h)     -> C00 -= 1, C01 += 1
        # M4 = d(g-e)     -> C00 += 1, C10 += 1
        # M5 = (a+b)h     -> C00 -= 1, C01 += 1
        # M6 = (c-a)(e+f) -> C11 += 1
        # M7 = (b-d)(g+h) -> C00 += 1, C11 -= 1
        
        # Note: C indices are (k, i).
        # C00 is C[k_off, i_off]
        # C01 is C[k_off, i_off+1]
        # C10 is C[k_off+1, i_off]
        # C11 is C[k_off+1, i_off+1]
        
        # M1
        U[i_off*m + j_off, col] = 1      # a
        U[(i_off+1)*m + (j_off+1), col] = 1 # d
        V[j_off*p + k_off, col] = 1      # e
        V[(j_off+1)*p + (k_off+1), col] = 1 # h
        W[k_off*n + i_off, col] = 1      # C00
        W[(k_off+1)*n + (i_off+1), col] = 1 # C11
        col += 1
        
        # M2
        U[i_off*m + (j_off+1), col] = 1      # c
        U[(i_off+1)*m + (j_off+1), col] = 1  # d
        V[j_off*p + k_off, col] = 1          # e
        W[(k_off+1)*n + i_off, col] = 1      # C10
        W[(k_off+1)*n + (i_off+1), col] = -1 # C11
        col += 1
        
        # M3
        U[i_off*m + j_off, col] = 1      # a
        V[(j_off)*p + (k_off+1), col] = 1 # f
        V[(j_off+1)*p + (k_off+1), col] = -1 # h
        W[k_off*n + i_off, col] = -1     # C00
        W[k_off*n + (i_off+1), col] = 1  # C01
        col += 1
        
        # M4
        U[(i_off+1)*m + (j_off+1), col] = 1 # d
        V[(j_off+1)*p + k_off, col] = 1     # g
        V[j_off*p + k_off, col] = -1        # e
        W[k_off*n + i_off, col] = 1         # C00
        W[(k_off+1)*n + i_off, col] = 1     # C10
        col += 1
        
        # M5
        U[i_off*m + j_off, col] = 1        # a
        U[i_off*m + (j_off+1), col] = 1    # b
        V[(j_off+1)*p + (k_off+1), col] = 1 # h
        W[k_off*n + i_off, col] = -1       # C00
        W[k_off*n + (i_off+1), col] = 1    # C01
        col += 1
        
        # M6
        U[i_off*m + (j_off+1), col] = 1    # c
        U[i_off*m + j_off, col] = -1       # a
        V[j_off*p + k_off, col] = 1        # e
        V[j_off*p + (k_off+1), col] = 1    # f
        W[(k_off+1)*n + (i_off+1), col] = 1 # C11
        col += 1
        
        # M7
        U[i_off*m + (j_off+1), col] = 1    # b
        U[(i_off+1)*m + (j_off+1), col] = -1 # d
        V[(j_off+1)*p + k_off, col] = 1    # g
        V[(j_off+1)*p + (k_off+1), col] = 1 # h
        W[k_off*n + i_off, col] = 1        # C00
        W[(k_off+1)*n + (i_off+1), col] = -1 # C11
        col += 1

    # Helper to add a 1x2x2 block
    # i_off is the row index in A (single row)
    # j_off, k_off are offsets for 2x2 block in B
    # C block is 2x1 at C[k_off:k_off+2, i_off]
    def add_1x2x2_block(i_off, j_off, k_off):
        nonlocal col
        # Term 1: A[i, j_off] * B[j_off, k_off:k_off+2] -> C[k_off:k_off+2, i]
        U[i_off*m + j_off, col] = 1
        V[j_off*p + k_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1
        
        # Term 2: A[i, j_off+1] * B[j_off+1, k_off:k_off+2] -> C[k_off:k_off+2, i]
        U[i_off*m + (j_off+1), col] = 1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1

    # Define blocks
    # A rows: 0,1 (2x2 blocks), 2 (1x2 blocks)
    # B cols: 0,1 (2x2), 2,3 (2x2)
    # C cols: 0,1 (2x2), 2 (1x2)
    # C rows: 0,1 (2x2), 2,3 (2x2)
    
    # 4 Strassen blocks (2x2x2)
    # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    # i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    # i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    # i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    
    # 4 1x2x2 blocks
    # i in {2}, j in {0,1}, k in {0,1}
    add_1x2x2_block(2, 0, 0)
    # i in {2}, j in {0,1}, k in {2,3}
    add_1x2x2_block(2, 0, 2)
    # i in {2}, j in {2,3}, k in {0,1}
    add_1x2x2_block(2, 2, 0)
    # i in {2}, j in {2,3}, k in {2,3}
    add_1x2x2_block(2, 2, 2)
    
    # Verify col is 36
    assert col == 36, f"Expected col 36, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

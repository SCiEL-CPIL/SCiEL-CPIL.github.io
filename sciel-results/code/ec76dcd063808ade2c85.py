import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 construction based on Strassen blocks for 2x2x2 and schoolbook for 1x4x6
    # Decomposition:
    # i in {0,1}: M_{2,4,6} -> 6 blocks of M_{2,2,2} (Rank 7 each) -> 42
    # i in {2}: M_{1,4,6} -> Rank 24
    # Total Rank = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global indices: i_g, j_g, k_g
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen 7 multiplications
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B01-B11)
        # m4 = A11(B10-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        
        # U entries for A
        # A00 is at (i_off, j_off)
        # A11 is at (i_off+1, j_off+1)
        # etc.
        
        # We need to map local (li, lj) to global (i, j)
        # U index: i*m + j
        
        # m1
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1       # C00 += m1
        W[(k_off+1)*n + (i_off+1)][col] = 1 # C11 += m1
        col += 1
        
        # m2
        U[i_off*m + (j_off+1)][col] = 1 # A10? No, A10 is (1,0). Local (1,0) -> i_off+1, j_off
        # Wait, Strassen indices:
        # A = [[A00, A01], [A10, A11]]
        # m2 = (A10+A11)B00
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        # C10 += m2
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # m3
        # m3 = A00(B01-B11)
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        # C01 += m3
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # m4
        # m4 = A11(B10-B00)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[j_off*p + k_off][col] = -1
        # C11 -= m4
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # m5
        # m5 = (A00+A01)B11
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        # C10 += m5
        W[k_off*n + (i_off+1)][col] = 1
        # C11 -= m5
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # m6
        # m6 = (A10-A00)(B00+B01)
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        # C10 += m6
        W[k_off*n + (i_off+1)][col] = 1
        # C11 += m6
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # m7
        # m7 = (A01-A11)(B10+B11)
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        # C01 += m7
        W[(k_off+1)*n + i_off][col] = 1
        # C00 -= m7
        W[k_off*n + i_off][col] = -1
        col += 1

    # Add 6 Strassen blocks for i in {0,1}
    # j in {0,1}, {2,3}
    # k in {0,1}, {2,3}, {4,5}
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            add_strassen_block(0, j_off, k_off)
            add_strassen_block(1, j_off, k_off) # Wait, i_off 0 and 1 are covered by the block?
            # No, a Strassen block covers i in {i_off, i_off+1}.
            # So add_strassen_block(0, j_off, k_off) covers i=0,1.
            # We don't need add_strassen_block(1, ...).
            # The block is 2x2x2.
            # So we just call it once per (j_off, k_off) pair for the i-pair (0,1).
            
    # Correction: The loop above calls add_strassen_block(0, ...) which handles i=0,1.
    # So we just need to iterate j_off and k_off.
    
    # Let's restart the column index logic to be safe
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            # Block for i in {0,1}
            add_strassen_block(0, j_off, k_off)
            
    # Now handle i in {2}
    # M_{1,4,6}
    # i=2
    # j in 0..3, k in 0..5
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

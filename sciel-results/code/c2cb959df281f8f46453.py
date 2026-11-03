"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Rank 50 decomposition using Strassen for 2x2x2 blocks and schoolbook for others.
    # Structure:
    # A (2x3) split into A1 (2x2) and A2 (2x1)
    # B (3x9) split into B1 (2x9) and B2 (1x9)
    # B1 split into four 2x2 blocks and one 2x1 block
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to set Strassen 2x2x2 block
    # A rows: i in {0,1}, j in {j_start, j_start+1}
    # B rows: j in {j_start, j_start+1}, k in {k_start, k_start+1}
    # C rows: k in {k_start, k_start+1}, i in {0,1}
    def set_strassen_2x2x2(j_start, k_start):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        U[j_start][col] = 1       # A00 (i=0, j=j_start)
        U[j_start+1][col] = 1     # A11 (i=1, j=j_start+1)
        V[j_start * p + k_start][col] = 1       # B00 (j=j_start, k=k_start)
        V[(j_start+1) * p + k_start+1][col] = 1 # B11 (j=j_start+1, k=k_start+1)
        W[k_start * n + 0][col] = 1             # C00 (k=k_start, i=0)
        W[(k_start+1) * n + 1][col] = 1         # C11 (k=k_start+1, i=1)
        col += 1
        
        # M2: (A10+A11)B00 -> C10 - C11
        U[j_start+1][col] = 1     # A10 (i=1, j=j_start)
        U[j_start+2][col] = 1     # A11 (i=1, j=j_start+1) -- Wait, A is 2x2.
        # A indices for 2x2 block:
        # i=0,1. j=j_start, j_start+1.
        # Global U index: i*m + j.
        # A10 is i=1, j=j_start. Index: 1*m + j_start.
        # A11 is i=1, j=j_start+1. Index: 1*m + j_start+1.
        
        # Correction:
        # In previous thought process I used U[3], U[4] for i=1, j=0,1 when m=3.
        # Here m=3.
        # A1 (2x2) corresponds to j=0,1.
        # i=0 -> rows 0,1. i=1 -> rows 3,4.
        # So A10 is U[3], A11 is U[4].
        
        # Let's re-verify indices.
        # n=2, m=3, p=9.
        # U shape 6xR. Index i*m+j.
        # A1 is cols 0,1 of A.
        # A10: i=1, j=0 -> 1*3+0=3.
        # A11: i=1, j=1 -> 1*3+1=4.
        # A00: i=0, j=0 -> 0.
        # A01: i=0, j=1 -> 1.
        
        # V shape 27xR. Index j*p+k.
        # B1 is rows 0,1 of B.
        # B00: j=0, k=k_start -> 0*9+k_start.
        # B11: j=1, k=k_start+1 -> 1*9+k_start+1.
        
        # W shape 18xR. Index k*n+i.
        # C00: k=k_start, i=0 -> k_start*2+0.
        # C11: k=k_start+1, i=1 -> (k_start+1)*2+1.
        
        # M2: (A10+A11)B00
        U[3][col] = 1
        U[4][col] = 1
        V[j_start * p + k_start][col] = 1 # B00
        W[(k_start) * n + 1][col] = 1     # C10 (k=k_start, i=1)
        W[(k_start+1) * n + 1][col] = -1  # C11 (k=k_start+1, i=1)
        col += 1
        
        # M3: A00(B01-B11)
        U[0][col] = 1 # A00
        V[j_start * p + k_start+1][col] = 1   # B01
        V[(j_start+1) * p + k_start+1][col] = -1 # B11
        W[(k_start+1) * n + 0][col] = 1      # C01
        W[(k_start+1) * n + 1][col] = 1      # C11
        col += 1
        
        # M4: A11(B10-B00)
        U[4][col] = 1 # A11
        V[(j_start+1) * p + k_start][col] = 1   # B10
        V[j_start * p + k_start][col] = -1      # B00
        W[(k_start) * n + 0][col] = 1           # C00
        W[(k_start) * n + 1][col] = 1           # C10
        col += 1
        
        # M5: (A00+A01)B11
        U[0][col] = 1 # A00
        U[1][col] = 1 # A01
        V[(j_start+1) * p + k_start+1][col] = 1 # B11
        W[(k_start) * n + 0][col] = -1          # C00
        W[(k_start+1) * n + 0][col] = 1         # C01
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[3][col] = 1  # A10
        U[0][col] = -1 # A00
        V[j_start * p + k_start][col] = 1   # B00
        V[j_start * p + k_start+1][col] = 1 # B01
        W[(k_start+1) * n + 1][col] = 1     # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1][col] = 1  # A01
        U[4][col] = -1 # A11
        V[(j_start+1) * p + k_start][col] = 1   # B10
        V[(j_start+1) * p + k_start+1][col] = 1 # B11
        W[(k_start) * n + 0][col] = 1           # C00
        col += 1

    # Block 1: A2 B2 (2x1x9)
    # A2 is i in {0,1}, j=2.
    # B2 is j=2, k in {0..8}.
    for k in range(9):
        # M_k1: A20 * B2k -> C_k0
        U[0 * m + 2][col] = 1 # A20 (i=0, j=2)
        V[2 * p + k][col] = 1 # B2k (j=2, k)
        W[k * n + 0][col] = 1 # C_k0 (k, i=0)
        col += 1
        
    for k in range(9):
        # M_k2: A21 * B2k -> C_k1
        U[1 * m + 2][col] = 1 # A21 (i=1, j=2)
        V[2 * p + k][col] = 1 # B2k (j=2, k)
        W[k * n + 1][col] = 1 # C_k1 (k, i=1)
        col += 1
        
    # Blocks 2-5: A1 B1i (2x2x2) for i=1..4
    # A1 is j=0,1.
    # B1 is j=0,1.
    # B1i is k_start = (i-1)*2.
    for i in range(4):
        k_start = i * 2
        set_strassen_2x2x2(0, k_start) # j_start=0
        
    # Block 6: A1 B15 (2x2x1)
    # B15 is j=0,1, k=8.
    # M1: A00 B08 -> C_80
    U[0][col] = 1
    V[0 * p + 8][col] = 1
    W[8 * n + 0][col] = 1
    col += 1
    
    # M2: A01 B18 -> C_80
    U[1][col] = 1
    V[1 * p + 8][col] = 1
    W[8 * n + 0][col] = 1
    col += 1
    
    # M3: A10 B08 -> C_81
    U[3][col] = 1
    V[0 * p + 8][col] = 1
    W[8 * n + 1][col] = 1
    col += 1
    
    # M4: A11 B18 -> C_81
    U[4][col] = 1
    V[1 * p + 8][col] = 1
    W[8 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

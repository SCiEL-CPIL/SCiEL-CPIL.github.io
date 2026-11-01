import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # p = 15
    # We use 7 blocks of 2x2x2 (Strassen, rank 7) and 1 block of 2x2x1 (Schoolbook, rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for U: u = i*m + j = 2i + j
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 2
    # i=1,j=1 -> 3
    
    # Indices for V: v = j*p + k = 15j + k
    # For a block with global k offset k_off:
    # local k=0 -> global k = k_off
    # local k=1 -> global k = k_off + 1
    # j=0: v = 15*0 + k = k
    # j=1: v = 15*1 + k = 15 + k
    
    # Indices for W: w = k*n + i = 2k + i
    # For a block with global k offset k_off:
    # local k=0 -> global k = k_off
    # local k=1 -> global k = k_off + 1
    # i=0: w = 2*k + 0 = 2k
    # i=1: w = 2*k + 1 = 2k + 1
    
    def add_strassen_block(k_off, start_col):
        # 7 terms
        # Each term adds 1 to col
        # Term 1: M1
        U[0][start_col] = 1
        U[3][start_col] = 1
        V[k_off][start_col] = 1       # j=0, k=k_off
        V[15 + k_off + 1][start_col] = 1 # j=1, k=k_off+1
        W[2 * k_off][start_col] = 1     # k=k_off, i=0
        W[2 * (k_off + 1) + 1][start_col] = 1 # k=k_off+1, i=1
        col = start_col + 1
        
        # Term 2: M2
        U[2][col] = 1 # A10
        U[3][col] = 1 # A11
        V[k_off][col] = 1 # B00 (j=0, k=k_off)
        W[2 * k_off + 1][col] = 1 # C10 (k=k_off, i=1)
        col += 1
        
        # Term 3: M3
        U[0][col] = 1 # A00
        V[15 + k_off + 1][col] = 1 # B11 (j=1, k=k_off+1)
        V[k_off + 1][col] = -1 # B01 (j=0, k=k_off+1)
        W[2 * (k_off + 1)][col] = 1 # C01 (k=k_off+1, i=0)
        col += 1
        
        # Term 4: M4
        U[3][col] = 1 # A11
        V[15 + k_off][col] = 1 # B10 (j=1, k=k_off)
        V[k_off][col] = -1 # B00 (j=0, k=k_off)
        W[2 * k_off][col] = 1 # C00 (k=k_off, i=0)
        W[2 * k_off + 1][col] = 1 # C10 (k=k_off, i=1)
        col += 1
        
        # Term 5: M5
        U[0][col] = 1 # A00
        U[1][col] = 1 # A01
        V[15 + k_off + 1][col] = 1 # B11 (j=1, k=k_off+1)
        W[2 * k_off][col] = -1 # C00 (k=k_off, i=0)
        W[2 * (k_off + 1)][col] = 1 # C01 (k=k_off+1, i=0)
        col += 1
        
        # Term 6: M6
        U[2][col] = 1 # A10
        U[0][col] = -1 # A00
        V[k_off][col] = 1 # B00 (j=0, k=k_off)
        V[k_off + 1][col] = 1 # B01 (j=0, k=k_off+1)
        W[2 * (k_off + 1) + 1][col] = 1 # C11 (k=k_off+1, i=1)
        col += 1
        
        # Term 7: M7
        U[1][col] = 1 # A01
        U[3][col] = -1 # A11
        V[15 + k_off][col] = 1 # B10 (j=1, k=k_off)
        V[15 + k_off + 1][col] = 1 # B11 (j=1, k=k_off+1)
        W[2 * k_off][col] = 1 # C00 (k=k_off, i=0)
        W[2 * (k_off + 1) + 1][col] = 1 # C11 (k=k_off+1, i=1)
        col += 1
        
        return col

    def add_schoolbook_block(k_off, start_col):
        # 2x2x1 block
        # C00 = A00 B00
        # C01 = A01 B00
        # C10 = A10 B00
        # C11 = A11 B00
        # Wait, for 2x2x1, B is 2x1. So B00 and B10.
        # C00 = A00 B00 + A01 B10 ? No.
        # C_i = sum_j A_ij B_j.
        # C0 = A00 B0 + A01 B1.
        # C1 = A10 B0 + A11 B1.
        
        # Term 1: A00 B0
        U[0][start_col] = 1
        V[k_off][start_col] = 1 # j=0, k=k_off
        W[2 * k_off][start_col] = 1 # k=k_off, i=0
        col = start_col + 1
        
        # Term 2: A01 B1
        U[1][col] = 1
        V[15 + k_off][col] = 1 # j=1, k=k_off
        W[2 * k_off][col] = 1 # k=k_off, i=0
        col += 1
        
        # Term 3: A10 B0
        U[2][col] = 1
        V[k_off][col] = 1 # j=0, k=k_off
        W[2 * k_off + 1][col] = 1 # k=k_off, i=1
        col += 1
        
        # Term 4: A11 B1
        U[3][col] = 1
        V[15 + k_off][col] = 1 # j=1, k=k_off
        W[2 * k_off + 1][col] = 1 # k=k_off, i=1
        col += 1
        
        return col

    # 7 blocks of 2
    for b in range(7):
        k_off = 2 * b
        col = add_strassen_block(k_off, col)
        
    # 1 block of 1
    k_off = 14
    col = add_schoolbook_block(k_off, col)
    
    return {"U": U, "V": V, "W": W}

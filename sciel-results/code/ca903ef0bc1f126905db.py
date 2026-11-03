import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x5) into B1 (2x5) and B2 (2x5)
    # C = A1 B1 + A2 B2
    # Each 2x2 x 2x5 multiplication has rank 18 (7+7+4)
    # Total rank 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 block
    # A indices: (i, j) where i in 0..1, j in 0..1 (local)
    # B indices: (j, k) where j in 0..1, k in 0..1 (local)
    # C indices: (i, k) where i in 0..1, k in 0..1 (local)
    # Global indices:
    # A: i_global = i_base + i, j_global = j_base + j
    # B: j_global = j_base + j, k_global = k_base + k
    # C: i_global = i_base + i, k_global = k_base + k
    # W index: k_global * n + i_global
    
    def add_strassen_2x2_2x2(i_base, j_base, k_base, start_col):
        # Strassen factors for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U contributions (A side)
        # A00: M1, M3, M5, M6(-)
        # A01: M5, M7(-)
        # A10: M2, M6
        # A11: M1, M2, M4, M7
        
        # Let's map local A indices (i,j) to global (i_base+i, j_base+j)
        # U[row, col]
        
        # M1
        U[i_base*2 + 0 + 0][start_col + 0] = 1 # A00
        U[i_base*2 + 1 + 1][start_col + 0] = 1 # A11
        V[(0 + 0)*5 + 0][start_col + 0] = 1 # B00 (j=0, k=0 relative to B block start? No, B is 2x5. j_base is 0 or 2. k_base is 0,2,4)
        # Wait, B indices in V are (j, k). j is 0..3, k is 0..4.
        # In this block, j is j_base..j_base+1, k is k_base..k_base+1.
        V[j_base*5 + k_base][start_col + 0] = 1
        V[(j_base+1)*5 + (k_base+1)][start_col + 0] = 1
        
        # W contributions (C side)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W index: k_global * n + i_global
        # C00: k=k_base, i=i_base -> W[k_base*2 + i_base]
        W[k_base*2 + i_base][start_col + 0] = 1
        # C11: k=k_base+1, i=i_base+1 -> W[(k_base+1)*2 + (i_base+1)]
        W[(k_base+1)*2 + (i_base+1)][start_col + 0] = 1
        
        # M2
        U[i_base*2 + 1 + 0][start_col + 1] = 1 # A10
        U[i_base*2 + 1 + 1][start_col + 1] = 1 # A11
        V[j_base*5 + k_base][start_col + 1] = 1 # B00
        # C10 = M2 + ... -> W[k_base*2 + (i_base+1)]
        W[k_base*2 + (i_base+1)][start_col + 1] = 1
        
        # M3
        U[i_base*2 + 0 + 0][start_col + 2] = 1 # A00
        V[(j_base+1)*5 + k_base][start_col + 2] = 1 # B10
        V[(j_base+1)*5 + (k_base+1)][start_col + 2] = -1 # B11
        # C01 = M3 + ... -> W[(k_base+1)*2 + i_base]
        W[(k_base+1)*2 + i_base][start_col + 2] = 1
        
        # M4
        U[i_base*2 + 1 + 1][start_col + 3] = 1 # A11
        V[j_base*5 + (k_base+1)][start_col + 3] = 1 # B01
        V[j_base*5 + k_base][start_col + 3] = -1 # B00
        # C00 = ... + M4 ...
        W[k_base*2 + i_base][start_col + 3] = 1
        # C10 = ... + M4 ...
        W[k_base*2 + (i_base+1)][start_col + 3] = 1
        
        # M5
        U[i_base*2 + 0 + 0][start_col + 4] = 1 # A00
        U[i_base*2 + 0 + 1][start_col + 4] = 1 # A01
        V[(j_base+1)*5 + (k_base+1)][start_col + 4] = 1 # B11
        # C00 = ... - M5 ...
        W[k_base*2 + i_base][start_col + 4] = -1
        # C01 = ... + M5 ...
        W[(k_base+1)*2 + i_base][start_col + 4] = 1
        
        # M6
        U[i_base*2 + 1 + 0][start_col + 5] = 1 # A10
        U[i_base*2 + 0 + 0][start_col + 5] = -1 # A00
        V[j_base*5 + k_base][start_col + 5] = 1 # B00
        V[j_base*5 + (k_base+1)][start_col + 5] = 1 # B01
        # C11 = ... + M6 ...
        W[(k_base+1)*2 + (i_base+1)][start_col + 5] = 1
        
        # M7
        U[i_base*2 + 0 + 1][start_col + 6] = 1 # A01
        U[i_base*2 + 1 + 1][start_col + 6] = -1 # A11
        V[(j_base+1)*5 + k_base][start_col + 6] = 1 # B10
        V[(j_base+1)*5 + (k_base+1)][start_col + 6] = 1 # B11
        # C00 = ... + M7 ...
        W[k_base*2 + i_base][start_col + 6] = 1
        
        return start_col + 7

    def add_schoolbook_2x2_2x1(i_base, j_base, k_base, start_col):
        # A (2x2) * B (2x1) -> C (2x1)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # 4 terms
        # Term 0: A00 B0 -> C0
        U[i_base*2 + 0][start_col + 0] = 1
        V[j_base*5 + k_base][start_col + 0] = 1
        W[k_base*2 + i_base][start_col + 0] = 1
        
        # Term 1: A01 B1 -> C0
        U[i_base*2 + 1][start_col + 1] = 1
        V[(j_base+1)*5 + (k_base+1)][start_col + 1] = 1
        W[k_base*2 + i_base][start_col + 1] = 1
        
        # Term 2: A10 B0 -> C1
        U[i_base*2 + 2][start_col + 2] = 1
        V[j_base*5 + k_base][start_col + 2] = 1
        W[k_base*2 + (i_base+1)][start_col + 2] = 1
        
        # Term 3: A11 B1 -> C1
        U[i_base*2 + 3][start_col + 3] = 1
        V[(j_base+1)*5 + (k_base+1)][start_col + 3] = 1
        W[k_base*2 + (i_base+1)][start_col + 3] = 1
        
        return start_col + 4

    col = 0
    # Block 1: A1 (i_base=0, j_base=0) * B1 (j_base=0)
    # B1 is 2x5. Split into 2x2, 2x2, 2x1.
    # k_base for B1: 0, 2, 4
    col = add_strassen_2x2_2x2(0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 0, 2, col)
    col = add_schoolbook_2x2_2x1(0, 0, 4, col)
    
    # Block 2: A2 (i_base=0, j_base=2) * B2 (j_base=2)
    # B2 is 2x5. Split into 2x2, 2x2, 2x1.
    # k_base for B2: 0, 2, 4
    col = add_strassen_2x2_2x2(0, 2, 0, col)
    col = add_strassen_2x2_2x2(0, 2, 2, col)
    col = add_schoolbook_2x2_2x1(0, 2, 4, col)
    
    return {"U": U, "V": V, "W": W}
